export @metal


## high-level @metal interface

const MACRO_KWARGS = [:launch]
const COMPILER_KWARGS = [:kernel, :name, :always_inline, :macos, :air, :metal]
const LAUNCH_KWARGS = [:groups, :threads, :queue]

"""
    @metal threads=... groups=... [kwargs...] func(args...)

High-level interface for executing code on a GPU.

The `@metal` macro should prefix a call, with `func` a callable function or object that
should return nothing. It will be compiled to a Metal function upon first use, and to a
certain extent arguments will be converted and managed automatically using `mtlconvert`.
Finally, a call to `mtlcall` is performed, creating a command buffer in the current global
command queue then committing it.

There is one supported keyword argument that influences the behavior of `@metal`:

- `launch`: whether to launch this kernel, defaults to `true`. If `false` the returned
  kernel object should be launched by calling it and passing arguments again.
- `name`: the name of the kernel in the generated code. Defaults to an automatically-
  generated name.
- `queue`: the command queue to use for this kernel. Defaults to the global command queue.
"""
macro metal(ex...)
    call = ex[end]
    kwargs = map(ex[1:end-1]) do kwarg
        if kwarg isa Symbol
            :($kwarg = $kwarg)
        elseif Meta.isexpr(kwarg, :(=))
            kwarg
        else
            throw(ArgumentError("Invalid keyword argument '$kwarg'"))
        end
    end

    # destructure the kernel call
    Meta.isexpr(call, :call) || throw(ArgumentError("second argument to @metal should be a function call"))
    f = call.args[1]
    args = call.args[2:end]

    code = quote end
    vars, var_exprs = assign_args!(code, args)

    # group keyword argument
    macro_kwargs, compiler_kwargs, call_kwargs, other_kwargs =
        split_kwargs(kwargs, MACRO_KWARGS, COMPILER_KWARGS, LAUNCH_KWARGS)
    if !isempty(other_kwargs)
        key,val = first(other_kwargs).args
        throw(ArgumentError("Unsupported keyword argument '$key'"))
    end

    # handle keyword arguments that influence the macro's behavior
    launch = true
    for kwarg in macro_kwargs
        key,val = kwarg.args
        if key === :launch
            isa(val, Bool) || throw(ArgumentError("`launch` keyword argument to @metal should be a Bool"))
            launch = val::Bool
        else
            throw(ArgumentError("Unsupported keyword argument '$key'"))
        end
    end
    if !launch && !isempty(call_kwargs)
        error("@metal with launch=false does not support launch-time keyword arguments; use them when calling the kernel")
    end

    # FIXME: macro hygiene wrt. escaping kwarg values (this broke with 1.5)
    #        we esc() the whole thing now, necessitating gensyms...
    @gensym f_var kernel_f kernel_args kernel_tt kernel

    # convert the arguments, call the compiler and launch the kernel
    # while keeping the original arguments alive
    push!(code.args,
        quote
            $f_var = $f
            GC.@preserve $(vars...) $f_var begin
                $kernel_f = $mtlconvert($f_var)
                $kernel_args = map($mtlconvert, ($(var_exprs...),))
                $kernel_tt = Tuple{map(Core.Typeof, $kernel_args)...}
                $kernel = $mtlfunction($kernel_f, $kernel_tt; $(compiler_kwargs...))
                if $launch
                    $kernel($(var_exprs...); $(call_kwargs...))
                end
                $kernel
            end
         end)

    return esc(quote
        let
            $code
        end
    end)
end


## argument conversion

struct Adaptor
    # the current command encoder, if any.
    cce::Union{Nothing,MTLComputeCommandEncoder}
end

# convert Metal buffers to their GPU address
function Adapt.adapt_storage(to::Adaptor, buf::MTLBuffer)
    if to.cce !== nothing
        MTL.use!(to.cce, buf, MTL.ReadWriteUsage)
    end
    reinterpret(Core.LLVMPtr{Nothing,AS.Device}, buf.gpuAddress)
end
function Adapt.adapt_storage(to::Adaptor, ptr::MtlPtr{T}) where {T}
    reinterpret(Core.LLVMPtr{T,AS.Device}, adapt(to, ptr.buffer)) + ptr.offset
end

# convert Metal host arrays to device arrays
function Adapt.adapt_storage(to::Adaptor, xs::MtlArray{T,N}) where {T,N}
    buf = pointer(xs)
    ptr = adapt(to, buf)
    MtlDeviceArray{T,N,AS.Device}(xs.dims, ptr)
end

# Base.RefValue isn't GPU compatible, so provide a compatible alternative
# TODO: port improvements from CUDA.jl
struct MtlRefValue{T} <: Ref{T}
  x::T
end
Base.getindex(r::MtlRefValue) = r.x
Adapt.adapt_structure(to::Adaptor, r::Base.RefValue) = MtlRefValue(adapt(to, r[]))

# broadcast sometimes passes a ref(type), resulting in a GPU-incompatible DataType box.
# avoid that by using a special kind of ref that knows about the boxed type.
struct MtlRefType{T} <: Ref{DataType} end
Base.getindex(r::MtlRefType{T}) where T = T
Adapt.adapt_structure(to::Adaptor, r::Base.RefValue{<:Union{DataType,Type}}) =
    MtlRefType{r[]}()

# case where type is the function being broadcasted
Adapt.adapt_structure(to::Adaptor,
                      bc::Broadcast.Broadcasted{Style, <:Any, Type{T}}) where {Style, T} =
    Broadcast.Broadcasted{Style}((x...) -> T(x...), adapt(to, bc.args), bc.axes)

"""
  mtlconvert(x, [cce])

This function is called for every argument to be passed to a kernel, allowing it to be
converted to a GPU-friendly format. By default, the function does nothing and returns the
input object `x` as-is.

Do not add methods to this function, but instead extend the underlying Adapt.jl package and
register methods for the the `Metal.Adaptor` type.
"""
mtlconvert(arg, cce=nothing) = adapt(Adaptor(cce), arg)


## host-side kernel API

struct HostKernel{F,TT}
    f::F
    pipeline::MTLComputePipelineState
    loggingEnabled::Bool
end

const mtlfunction_lock = ReentrantLock()

"""
    mtlfunction(f, tt=Tuple{}; kwargs...)

Low-level interface to compile a function invocation for the currently-active GPU, returning
a callable kernel object. For a higher-level interface, use [`@metal`](@ref).

The following keyword arguments are supported:
- `macos`, `metal` and `air`: to override the macOS OS, Metal language and AIR bitcode
   versions used during compilation. Value should be a valid version number.

The output of this function is automatically cached, i.e. you can simply call `mtlfunction`
in a hot path without degrading performance. New code will be generated automatically when
the function changes, or when different types or keyword arguments are provided.
"""
function mtlfunction(f::F, tt::TT=Tuple{}; name=nothing, kwargs...) where {F,TT}
    dev = device()
    Base.@lock mtlfunction_lock begin
        # compile the function
        cache = compiler_cache(dev)
        source = methodinstance(F, tt)
        config = compiler_config(dev; name, kwargs...)::MetalCompilerConfig
        pipeline, loggingEnabled = GPUCompiler.cached_compilation(cache, source, config, compile, link)

        # create a callable object that captures the function instance. we don't need to think
        # about world age here, as GPUCompiler already does and will return a different object
        h = hash(pipeline, hash(f, hash(tt)))
        kernel = get(_kernel_instances, h, nothing)
        if kernel === nothing
            # create the kernel state object
            kernel = HostKernel{F,tt}(f, pipeline, loggingEnabled)
            _kernel_instances[h] = kernel
        end
        return kernel::HostKernel{F,tt}
    end
end

# cache of kernel instances
const _kernel_instances = Dict{UInt, Any}()


## kernel launching and argument encoding

@inline @generated function encode_arguments!(cce, kernel, args...)
    ex = quote
        bufs = MTLBuffer[]
    end

    # the arguments passed into this function have not been `mtlconvert`ed, because we need
    # to retain the top-level MTLBuffer and MtlPtr objects. eager conversion of nested
    # such objects to LLVMPtr seems fine, somehow.
    # TODO: can we just convert everything eagerly and support top-level LLVMPtrs?

    idx = 1
    for (argidx, argtyp) in enumerate(args)
        argex = :(args[$argidx])
        if argtyp <: MTLBuffer
            # top-level buffers are passed as a pointer-valued argument
            push!(ex.args, :(set_buffer!(cce, $argex, 0, $idx)))
        elseif argtyp <: MtlPtr
            # the same as a buffer, but with an offset
            push!(ex.args, :(set_buffer!(cce, $argex.buffer, $argex.offset, $idx)))
        elseif isghosttype(argtyp) || Core.Compiler.isconstType(argtyp)
            continue
        else
            # everything else is passed by reference, in an argument buffer
            append!(ex.args, (quote
                buf = encode_argument!(kernel, mtlconvert($(argex), cce))
                set_buffer!(cce, buf, 0, $idx)
                push!(bufs, buf)
            end).args)
        end
        idx += 1
    end

    append!(ex.args, (quote
        return bufs
    end).args)

    ex
end

@inline function encode_argument!(kernel, arg)
    argtyp = typeof(arg)

    # replace non-isbits arguments (they should be unused, or compilation
    # would have failed) by a dummy reference
    if !isbitstype(argtyp)
        arg = C_NULL
        argtyp = Ptr{Any}
    end

    # pass by reference, in an argument buffer
    argument_buffer = alloc(kernel.pipeline.device, sizeof(argtyp); storage=SharedStorage)
    argument_buffer.label = "MTLBuffer for kernel argument"
    unsafe_store!(convert(Ptr{argtyp}, argument_buffer), arg)
    return argument_buffer
end

@autoreleasepool function (kernel::HostKernel)(args...; groups=1, threads=1,
                                               queue=global_queue(device()))
    groups = MTLSize(groups)
    threads = MTLSize(threads)
    (groups.width>0 && groups.height>0 && groups.depth>0) ||
        throw(ArgumentError("All group dimensions should be non-zero"))
    (threads.width>0 && threads.height>0 && threads.depth>0) ||
        throw(ArgumentError("All thread dimensions should be non-zero"))

    (threads.width * threads.height * threads.depth) > kernel.pipeline.maxTotalThreadsPerThreadgroup &&
        throw(ArgumentError("Number of threads in group ($(threads.width * threads.height * threads.depth)) should not exceed $(kernel.pipeline.maxTotalThreadsPerThreadgroup)"))

    cmdbuf = if kernel.loggingEnabled
        # TODO: make this a dynamic error, i.e., from the kernel (JuliaGPU/Metal.jl#433)
        if macos_version() < v"15"
            error("Logging is only supported on macOS 15 or higher")
        end

        if MTLCaptureManager().isCapturing
            error("Logging is not supported while GPU frame capturing")
        end

        log_state_descriptor = MTLLogStateDescriptor()
        log_state_descriptor.level = MTL.MTLLogLevelDebug
        log_state = MTLLogState(queue.device, log_state_descriptor)

        function log_handler(subSystem, category, logLevel, message)
            Core.print(String(NSString(message)))
            return nothing
        end

        block = @objcblock(log_handler, Nothing, (id{NSString}, id{NSString}, NSInteger, id{NSString}))
        @objc [log_state::id{MTLLogState} addLogHandler:block::id{NSBlock}]::Nothing

        cmdbuf_descriptor = MTLCommandBufferDescriptor()
        cmdbuf_descriptor.logState = log_state
        MTLCommandBuffer(queue, cmdbuf_descriptor)
    else
        MTLCommandBuffer(queue)
    end

    cmdbuf.label = "MTLCommandBuffer($(nameof(kernel.f)))"
    cce = MTLComputeCommandEncoder(cmdbuf)
    argument_buffers = try
        MTL.set_function!(cce, kernel.pipeline)
        bufs = encode_arguments!(cce, kernel, kernel.f, args...)
        MTL.append_current_function!(cce, groups, threads)
        bufs
    finally
        close(cce)
    end

    # the command buffer retains resources that are explicitly encoded (i.e. direct buffer
    # arguments, or the buffers allocated for each other argument), but that doesn't keep
    # other resources alive for which we've encoded the GPU address ourselves. since it's
    # possible for buffers to go out of scope while the kernel is still running, which
    # triggers validation failures, keep track of things we need to keep alive until the
    # kernel has actually completed.
    #
    # TODO: is there a way to bind additional resources to the command buffer?
    roots = [kernel.f, args]
    MTL.on_completed(cmdbuf) do buf
        empty!(roots)
        foreach(free, argument_buffers)

        # Check for errors
        # XXX: we cannot do this nicely, e.g. throwing an `error` or reporting with `@error`
        #      because we're not allowed to switch tasks from this contexts.
        if buf.status == MTL.MTLCommandBufferStatusError
            Core.println("ERROR: Failed to submit command buffer: $(buf.error.localizedDescription)")
        end

    end
    commit!(cmdbuf)
end

## Intra-warp Helpers

"""
    nextwarp(dev, threads)
    prevwarp(dev, threads)

Returns the next or previous nearest number of threads that is a multiple of the warp size
of a device `dev`. This is a common requirement when using intra-warp communication.
"""
function nextwarp(pipe::MTLComputePipelineState, threads::Integer)
    ws = pipe.threadExecutionWidth
    return threads + (ws - threads % ws) % ws
end

@doc (@doc nextwarp) function prevwarp(pipe::MTLComputePipelineState, threads::Integer)
    ws = pipe.threadExecutionWidth
    return threads - Base.rem(threads, ws)
end
