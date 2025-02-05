export device, device!, global_queue, synchronize, device_synchronize

log_compiler()          = OSLog("org.juliagpu.metal", "Compiler")
log_compiler(args...)   = log_compiler()(args...)
log_array()             = OSLog("org.juliagpu.metal", "Array")
log_array(args...)      = log_array()(args...)

"""
    device()::MTLDevice

Return the Metal GPU device associated with the current Julia task.

Since all M-series systems currently only externally show a single GPU, this function
effectively returns the only system GPU.
"""
function device()
    get!(task_local_storage(), :MTLDevice) do
        dev = MTLDevice(1)
        if !supports_family(dev, MTL.MTLGPUFamilyApple7)
            @warn """Metal.jl is only supported on non-virtualized Apple Silicon, you may run into issues.
                     See https://github.com/JuliaGPU/Metal.jl/issues/22 for more details.""" maxlog=1
        end
        if !supports_family(dev, MTL.MTLGPUFamilyMetal3)
            @warn """Metal.jl is only supported on Metal 3-capable devices, you may run into issues.
                     See https://github.com/JuliaGPU/Metal.jl/issues/22 for more details.""" maxlog=1
        end
        return dev
    end::MTLDevice
end

"""
    device!(dev::MTLDevice)

Sets the Metal GPU device associated with the current Julia task.
"""
device!(dev::MTLDevice) = task_local_storage(:MTLDevice, dev)

const global_queues = WeakKeyDict{MTLCommandQueue,Nothing}()

"""
    global_queue(dev::MTLDevice)::MTLCommandQueue

Return the Metal command queue associated with the current Julia thread.
"""
function global_queue(dev::MTLDevice)
    get!(task_local_storage(), (:MTLCommandQueue, dev)) do
        @autoreleasepool begin
            # NOTE: MTLCommandQueue itself is manually reference-counted,
            #       the release pool is for resources used during its construction.

            queue = MTLCommandQueue(dev)
            queue.label = "global_queue($(current_task()))"
            global_queues[queue] = nothing
            queue
        end
    end::MTLCommandQueue
end

# TODO: Increase performance (currently ~15us)
"""
    synchronize(queue)

Wait for currently committed GPU work on this queue to finish.

Create a new MTLCommandBuffer from the global command queue, commit it to the queue,
and simply wait for it to be completed. Since command buffers *should* execute in a
First-In-First-Out manner, this synchronizes the GPU.
"""
@autoreleasepool function synchronize(queue::MTLCommandQueue=global_queue(device()))
    cmdbuf = MTLCommandBuffer(queue)
    commit!(cmdbuf)
    wait_completed(cmdbuf)
end

"""
    device_synchronize()

Synchronize all committed GPU work across all global queues
"""
function device_synchronize()
    for queue in keys(global_queues)
        synchronize(queue)
    end
end
