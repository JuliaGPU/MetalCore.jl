export MTLLogLevel

export MTLLogStateDescriptor

# @objcwrapper immutable = true MTLLogStateDescriptor <: NSObject

function MTLLogStateDescriptor()
    handle = @objc [MTLLogStateDescriptor alloc]::id{MTLLogStateDescriptor}
    obj = MTLLogStateDescriptor(handle)
    finalizer(release, obj)
    @objc [obj::id{MTLLogStateDescriptor} init]::id{MTLLogStateDescriptor}
    return obj
end


export MTLLogState

# @objcwrapper immutable = true MTLLogState <: NSObject

function MTLLogState(dev::MTLDevice, descriptor::MTLLogStateDescriptor)
    err = Ref{id{NSError}}(nil)
    handle = @objc [dev::id{MTLDevice} newLogStateWithDescriptor:descriptor::id{MTLLogStateDescriptor}
                                       error:err::Ptr{id{NSError}}]::id{MTLLogState}
    err[] == nil || throw(NSError(err[]))
    MTLLogState(handle)
end
