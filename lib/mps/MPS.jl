"""
# MPS

`MPS` is where the Metal Performance Shaders API wrappers are defined.

Not all functionality is currently implemented or documented. For further details,
refer to the [official Apple documentation](https://developer.apple.com/documentation/metalperformanceshaders).
"""
module MPS

using ..Metal
using .MTL

using CEnum
using ObjectiveC, .Foundation

import GPUArrays

using BFloat16s

const MtlFloat = Union{Float32, Float16}

const MPSShape = NSArray#{NSNumber}
Base.convert(::Type{MPSShape}, tuple::Union{Vector{N},NTuple{N, <:Integer}}) where N = NSArray(NSNumber.(collect(tuple)))

is_supported(dev::MTLDevice) = ccall(:MPSSupportsMTLDevice, Bool, (id{MTLDevice},), dev)

# Load in generated enums and structs
include("libmps.jl")

include("size.jl")

# high-level wrappers
include("command_buf.jl")
include("kernel.jl")
include("images.jl")
include("matrix.jl")
include("vector.jl")
include("matrixrandom.jl")
include("ndarray.jl")
include("decomposition.jl")
include("copy.jl")

# integrations
include("random.jl")
include("linalg.jl")

end
