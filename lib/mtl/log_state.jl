export MTLLogLevel

@cenum MTLLogLevel::NSInteger begin
    MTLLogLevelUndefined = 0
    MTLLogLevelDebug = 1
    MTLLogLevelInfo = 2
    MTLLogLevelNotice = 3
    MTLLogLevelError = 4
    MTLLogLevelFault = 5
end

export MTLLogStateDescriptor

@objcwrapper immutable=false MTLLogStateDescriptor <: NSObject

@objcproperties MTLLogStateDescriptor begin
    @autoproperty level::MTLLogLevel setter=setLevel
    @autoproperty bufferSize::NSInteger setter=setBufferSize
end

function MTLLogStateDescriptor()
    handle = @objc [MTLLogStateDescriptor alloc]::id{MTLLogStateDescriptor}
    obj = MTLLogStateDescriptor(handle)
    finalizer(release, obj)
    @objc [obj::id{MTLLogStateDescriptor} init]::id{MTLLogStateDescriptor}
    return obj
end


export MTLLogState

@objcwrapper MTLLogState <: NSObject

function MTLLogState(dev::MTLDevice, descriptor::MTLLogStateDescriptor)
    err = Ref{id{NSError}}(nil)
    handle = @objc [dev::id{MTLDevice} newLogStateWithDescriptor:descriptor::id{MTLLogStateDescriptor}
                                       error:err::Ptr{id{NSError}}]::id{MTLLogState}
    err[] == nil || throw(NSError(err[]))
    MTLLogState(handle)
end
