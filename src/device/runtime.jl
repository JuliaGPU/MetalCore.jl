# GPU runtime library

# reset the runtime cache from global scope, so that any change triggers recompilation
GPUCompiler.reset_runtime()


## exception handling

struct ExceptionInfo_st
    # whether an exception has been encountered (0 -> 1)
    status::Int32

    # whether an exception is in the process of being reported (0 -> 1 -> 2)
    output_lock::Int32

    # who is reporting the exception
    thread::@NamedTuple{x::Int32,y::Int32,z::Int32}
    threadgroup::@NamedTuple{x::Int32,y::Int32,z::Int32}

    ExceptionInfo_st() = new(0, 0,
                             (; x=Int32(0), y=Int32(0), z=Int32(0)),
                             (; x=Int32(0), y=Int32(0), z=Int32(0)))
end


# to simplify use of this struct, which is passed by-reference, use property overloading
const ExceptionInfo = Ptr{ExceptionInfo_st}
@inline function Base.getproperty(info::ExceptionInfo, sym::Symbol)
    if sym === :status
        unsafe_load(convert(Ptr{Int32}, info))
    elseif sym === :output_lock
        # XXX: atomic_load_explicit?
        unsafe_load(convert(Ptr{Int32}, info + 4))
    elseif sym === :output_lock_ptr
        reinterpret(LLVMPtr{Int32,AS.Device}, info + 4)
    elseif sym === :thread
        unsafe_load(convert(Ptr{@NamedTuple{x::Int32,y::Int32,z::Int32}}, info + 8))
    elseif sym === :threadgroup
        unsafe_load(convert(Ptr{@NamedTuple{x::Int32,y::Int32,z::Int32}}, info + 20))
    else
        getfield(info, sym)
    end
end
@inline function Base.setproperty!(info::ExceptionInfo, sym::Symbol, value)
    if sym === :status
        unsafe_store!(convert(Ptr{Int32}, info), value)
    elseif sym === :output_lock
        # XXX: atomic_store_explicit?
        unsafe_store!(convert(Ptr{Int32}, info + 4), value)
    elseif sym === :thread
        unsafe_store!(convert(Ptr{@NamedTuple{x::Int32,y::Int32,z::Int32}}, info + 8), value)
    elseif sym === :threadgroup
        unsafe_store!(convert(Ptr{@NamedTuple{x::Int32,y::Int32,z::Int32}}, info + 20), value)
    else
        setfield!(info, sym, value)
    end
end

# it's not useful to have several threads report exceptions, so use an output
# lock to only have a single thread write an exception message
@inline function lock_output!(info::ExceptionInfo)
    if atomic_compare_exchange_weak_explicit(info.output_lock_ptr, Int32(0), Int32(1)) == Int32(0)
        # we just took the lock, so note our position
        info.thread, info.threadgroup = thread_position_in_threadgroup_3d(),
                                        threadgroup_position_in_grid_3d()
        #threadfence()
        return true
    elseif info.output_lock == 1 &&
           info.thread == thread_position_in_threadgroup_3d() &&
           info.threadgroup == threadgroup_position_in_grid_3d()
        # we already have the lock
        return true
    else
        # somebody else has the lock
        return false
    end
end

function report_exception_name(ex)
    info = kernel_state().exception_info

    # this is the first reporting function being called, so claim the exception
    if lock_output!(info)
        #@cuprintf("ERROR: a %s was thrown during kernel execution on thread (%d, %d, %d) in block (%d, %d, %d).\n",
        #          ex, threadIdx().x, threadIdx().y, threadIdx().z, blockIdx().x, blockIdx().y, blockIdx().z)
        #@cuprintf("Stacktrace:\n")
    end
    return
end

function report_exception_frame(idx, func, file, line)
    info = kernel_state().exception_info

    if lock_output!(info)
        #@cuprintf(" [%d] %s at %s:%d\n", idx, func, file, line)
    end
    return
end

function signal_exception()
    info = kernel_state().exception_info

    # finalize output
    if lock_output!(info)
        #@cuprintf("\n")
        info.output_lock = 2
    end

    # inform the host
    info.status = 1

    # XXX: threadgroup_barrier(MemoryFlagDevice) expects all threads to execute
    #      the barrier, so would deadlock
    #threadfence_system()

    # stop executing
    # XXX: we don't have a way to stop execution, so just return
    #      (GPUCompiler.jl will emit a trap instruction anyway)
    #exit()

    return
end


## kernel state

struct KernelState
    exception_info::ExceptionInfo
end

@inline @generated kernel_state() = GPUCompiler.kernel_state_value(KernelState)


## other

function report_oom(sz)
    #@cuprintf("ERROR: Out of dynamic GPU memory (trying to allocate %d bytes)\n", sz)
    return
end
