
@static if Metal.is_macos(v"15.0.0")
    export MTLCommandQueueDescriptor

    # @objcwrapper immutable=false MTLCommandQueueDescriptor <: NSObject

    function MTLCommandQueueDescriptor()
        handle = @objc [MTLCommandQueueDescriptor alloc]::id{MTLCommandQueueDescriptor}
        obj = MTLCommandQueueDescriptor(handle)
        finalizer(release, obj)
        @objc [obj::id{MTLCommandQueueDescriptor} init]::id{MTLCommandQueueDescriptor}
        return obj
    end

    function MTLCommandQueue(dev::MTLDevice, descriptor::MTLCommandQueueDescriptor)
        handle = @objc [dev::id{MTLDevice} newCommandQueueWithDescriptor:descriptor::id{MTLCommandQueueDescriptor}]::id{MTLCommandQueue}
        obj = MTLCommandQueue(handle)
        finalizer(release, obj)
        return obj
    end
end


export MTLCommandQueue

# @objcwrapper immutable=false MTLCommandQueue <: NSObject

function MTLCommandQueue(dev::MTLDevice)
    handle = @objc [dev::id{MTLDevice} newCommandQueue]::id{MTLCommandQueue}
    obj = MTLCommandQueue(handle)
    finalizer(release, obj)
    return obj
end

