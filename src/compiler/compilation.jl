## gpucompiler interface implementation

struct MetalCompilerParams <: AbstractCompilerParams end
const MetalCompilerConfig = CompilerConfig{MetalCompilerTarget, MetalCompilerParams}
const MetalCompilerJob = CompilerJob{MetalCompilerTarget, MetalCompilerParams}

GPUCompiler.runtime_module(::MetalCompilerJob) = Metal

const ci_cache = GPUCompiler.CodeCache()
GPUCompiler.ci_cache(::MetalCompilerJob) = ci_cache

GPUCompiler.method_table(::MetalCompilerJob) = method_table


## compiler implementation (cache, configure, compile, and link)

# cache of compilation caches, per device
const _compiler_caches = Dict{MTLDevice, Dict{UInt, Any}}()
function compiler_cache(ctx::MTLDevice)
    cache = get(_compiler_caches, ctx, nothing)
    if cache === nothing
        cache = Dict{UInt, Any}()
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
@noinline function _compiler_config(dev; kernel=true, name=nothing, always_inline=false, kwargs...)
    # TODO: configure the compiler target based on the device

    macos=macos_version()

    # create GPUCompiler objects
    target = MetalCompilerTarget(macos; kwargs...)
    params = MetalCompilerParams()
    CompilerConfig(target, params; kernel, name, always_inline)
end

# compile to executable machine code
function compile(@nospecialize(job::CompilerJob))
    # TODO: on 1.9, this actually creates a context. cache those.
    JuliaContext() do ctx
        compile(job, ctx)
    end
end
function compile(@nospecialize(job::CompilerJob), ctx)
    mi, mi_meta = GPUCompiler.emit_julia(job)
    ir, ir_meta = GPUCompiler.emit_llvm(job, mi; ctx)
    entry = LLVM.name(ir_meta.entry)
    image, asm_meta = GPUCompiler.emit_asm(job, ir; format=LLVM.API.LLVMObjectFile)

    return (; image, entry)
end

# link into an executable kernel
function link(@nospecialize(job::CompilerJob), compiled)
    dev = current_device()
    lib = MTLLibraryFromData(dev, compiled.image)
    fun = MTLFunction(lib, compiled.entry)
    pipeline = try
        MTLComputePipelineState(dev, fun)
    catch
        # the back-end compiler likely failed
        # XXX: check more accurately? the error domain doesn't help much here
        metallib = tempname(cleanup=false) * ".metallib"
        write(metallib, compiled.image)
        @warn """Compilation of MetalLib to native code failed.
                 If you think this is a bug, please file an issue and attach $(metallib)."""
        rethrow()
    end
    pipeline
end
