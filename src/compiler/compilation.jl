## gpucompiler interface implementation

struct MetalCompilerParams <: AbstractCompilerParams end
const MetalCompilerConfig = CompilerConfig{MetalCompilerTarget, MetalCompilerParams}
const MetalCompilerJob = CompilerJob{MetalCompilerTarget, MetalCompilerParams}

GPUCompiler.runtime_module(::MetalCompilerJob) = Metal

GPUCompiler.method_table(::MetalCompilerJob) = method_table


function GPUCompiler.finish_ir!(@nospecialize(job::MetalCompilerJob),
                                    mod::LLVM.Module, entry::LLVM.Function)
    entry = invoke(GPUCompiler.finish_ir!,
                   Tuple{CompilerJob{MetalCompilerTarget}, LLVM.Module, LLVM.Function},
                   job, mod, entry)

    # pointer type information for typed intrinsics
    # (this is consumed by the LLVM IR downgrader)
    for (jltyp, llvmtyp) in (Int32 => :i32, Int64 => :i64,
                             Float16 => :f16, Float32 => :f32),
        (as, asname) in (AS.Device => "global", AS.ThreadGroup => "local")

        # map of intrinsics to pointer operand indices and eltypes
        intrinsics = Dict()
        ## simd
        intrinsics["simdgroup_matrix_8x8_load.v64$llvmtyp.p$as$llvmtyp"] = (1 => jltyp,)
        intrinsics["simdgroup_matrix_8x8_store.v64$llvmtyp.p$as$llvmtyp"] = (2 => jltyp,)
        ## atomics
        for op in [:store, :load, :xchg, :add, :sub, :min, :max, :and, :or, :xor]
            intrinsics["atomic.$asname.$op.$llvmtyp"] = (1 => jltyp,)
        end
        intrinsics["atomic.$asname.cmpxchg.weak.$llvmtyp"] = (1 => jltyp, 2 => jltyp)

        # apply metadata to the function declarations
        for (intr, args) in intrinsics
            fn = "air.$intr"
            haskey(functions(mod), fn) || continue
            f = functions(mod)[fn]
            mds = []
            for (idx, typ) in args
                push!(mds, ConstantInt(Int32(idx-1)))
                push!(mds, null(convert(LLVMType, typ)))
            end
            metadata(f)["arg_eltypes"] = MDNode(mds)
        end
    end

    return entry
end


## compiler implementation (cache, configure, compile, and link)

# cache of compilation caches, per device
const _compiler_caches = Dict{MTLDevice, Dict{Any, Any}}()
function compiler_cache(ctx::MTLDevice)
    cache = get(_compiler_caches, ctx, nothing)
    if cache === nothing
        cache = Dict{Any, Any}()
        _compiler_caches[ctx] = cache
    end
    return cache
end

# cache of compiler configurations, per device (but additionally configurable via kwargs)
const _toolchain = Ref{Any}()
const _compiler_configs = Dict{UInt, MetalCompilerConfig}()
function compiler_config(dev; kwargs...)
    h = hash(dev, hash(kwargs))
    config = get(_compiler_configs, h, nothing)
    if config === nothing
        config = _compiler_config(dev; kwargs...)
        _compiler_configs[h] = config
    end
    return config
end
@noinline function _compiler_config(dev; kernel=true, name=nothing, always_inline=false,
                                         macos=nothing, air=nothing, metal=nothing,
                                         kwargs...)
    # determine the versions of things to target
    if macos === nothing
        macos = macos_version()
    end
    if metal === nothing
        metal = metal_support()
    end
    if air === nothing
        # we support down to macOS 13, which supports AIR 2.5
        # so always target that version for now
        air = v"2.5"
        @assert air <= air_support()
    end

    # create GPUCompiler objects
    target = MetalCompilerTarget(; macos, air, metal, kwargs...)
    params = MetalCompilerParams()
    CompilerConfig(target, params; kernel, name, always_inline)
end

# compile to executable machine code
function compile(@nospecialize(job::CompilerJob))
    @signpost_event log=log_compiler() "Compile" "Job=$job"

    @signpost_interval log=log_compiler() "Generate LLVM IR" begin
        # TODO: on 1.9, this actually creates a context. cache those.
        ir, entry, loggingEnabled = JuliaContext() do ctx
            mod, meta = GPUCompiler.compile(:llvm, job)
            string(mod), LLVM.name(meta.entry), haskey(functions(mod), "air.os_log")
        end
    end

    @signpost_interval log=log_compiler() "Downgrade to AIR" begin
        # generate AIR
        air = let
            input = Pipe()
            output = Pipe()
            log = Pipe()

            cmd = `$(LLVMDowngrader_jll.llvm_as()) --bitcode-version=5.0 -o -`
            proc = run(pipeline(cmd, stdout=output, stderr=log, stdin=input); wait=false)
            close(output.in)
            close(log.in)

            writer = @async begin
                write(input, ir)
                close(input)
            end
            reader = @async read(output)
            logger = @async read(log, String)

            try
                wait(proc)
                success(proc) || error(fetch(logger))
            catch err
                ir_file = tempname(cleanup=false) * ".ll"
                write(ir_file, ir)
                if parse(Bool, get(ENV, "BUILDKITE", "false"))
                    run(`buildkite-agent artifact upload $(ir_file)`)
                end
                error("""Compilation to AIR failed; see above for details.
                         If you think this is a bug, please file an issue and attach $(ir_file)""")
            end

            fetch(reader)
        end
    end

    @signpost_interval log=log_compiler() "Create Metal library" begin
        metallib = try
            fun = MetalLibFunction(; name=entry, air_module=air,
                                     air_version=job.config.target.air,
                                     metal_version=job.config.target.metal)
            lib = MetalLib(; functions = [fun])

            io = IOBuffer()
            write(io, lib)
            take!(io)
        catch err
            ir_file = tempname(cleanup=false) * ".ll"
            write(ir_file, ir)
            air_file = tempname(cleanup=false) * ".air"
            write(air_file, air)
            if parse(Bool, get(ENV, "BUILDKITE", "false"))
                run(`buildkite-agent artifact upload $(ir_file)`)
                run(`buildkite-agent artifact upload $(air_file)`)
            end
            error("""Compilation to Metal library failed; see below for details.
                     If you think this is a bug, please file an issue and attach the following files:
                     - $(ir_file)
                     - $(air_file)""")
        end
    end

    return (; ir, air, metallib, entry, loggingEnabled)
end

# link into an executable kernel
@autoreleasepool function link(@nospecialize(job::CompilerJob), compiled)
    @signpost_event log=log_compiler() "Link" "Job=$job"

    @signpost_interval log=log_compiler() "Instantiate compute pipeline" begin
        dev = device()
        lib = MTLLibraryFromData(dev, compiled.metallib)
        fun = MTLFunction(lib, compiled.entry)
        pipeline_state = try
            MTLComputePipelineState(dev, fun)
        catch err
            isa(err, NSError) || rethrow()
            retain(err)

            # the back-end compiler likely failed
            # XXX: check more accurately? the error domain doesn't help much here
            ir_file = tempname(cleanup=false) * ".ll"
            write(ir_file, compiled.ir)
            air_file = tempname(cleanup=false) * ".air"
            write(air_file, compiled.air)
            metallib_file = tempname(cleanup=false) * ".metallib"
            write(metallib_file, compiled.metallib)
            if parse(Bool, get(ENV, "BUILDKITE", "false"))
                run(`buildkite-agent artifact upload $(ir_file)`)
                run(`buildkite-agent artifact upload $(air_file)`)
                run(`buildkite-agent artifact upload $(metallib_file)`)
            end
            error("""Compilation to native code failed; see below for details.
                     If you think this is a bug, please file an issue and attach the following files:
                     - $(ir_file)
                     - $(air_file)
                     - $(metallib_file)""")
        end
    end

    # most of the time, we don't need the function object,
    # so don't keep it alive unconditionally in GPUCompiler's caches
    pipeline_state, compiled.loggingEnabled
end
