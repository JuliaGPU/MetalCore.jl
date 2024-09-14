"""
# MTL

`MTL` is where the Metal API wrappers are defined.

Not all functionality is currently implemented or documented. For further details,
refer to the [official Apple documentation](https://developer.apple.com/documentation/metal).
"""
module MTL

using CEnum
using ObjectiveC, .Foundation, .Dispatch

using ..Metal

# Metal APIs generally expect to be running under an autorelease pool.
# In most cases, we handle this in the code calling into the MTL module,
# however, finalizers are out of the caller's control, so we need to
# ensure here already that they are running under an autorelease pool.
release(obj) = @autoreleasepool unsafe=true Foundation.release(obj)


## source code includes

include("libmtl.jl")
include("size.jl")
include("device.jl")
include("resource.jl")
include("storage_type.jl")
include("compile-opts.jl")
include("library.jl")
include("function.jl")
include("events.jl")
include("fences.jl")
include("heap.jl")
include("buffer.jl")
include("log_state.jl")
include("command_queue.jl")
include("command_buf.jl")
include("compute_pipeline.jl")
include("command_enc.jl")
include("command_enc/blit.jl")
include("command_enc/compute.jl")
include("binary_archive.jl")
include("capture.jl")
include("texture.jl")

end # module
