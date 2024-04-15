module MPS

using ..Metal

using CEnum
using ObjectiveC, .Foundation

import GPUArrays

const MtlFloat = Union{Float32, Float16}

is_supported(dev::MTLDevice) = ccall(:MPSSupportsMTLDevice, Bool, (id{MTLDevice},), dev)

include("size.jl")

# MPS kernel base clases
include("kernel.jl")
include("images.jl")

# high-level wrappers
include("matrix.jl")
include("vector.jl")

# integrations
include("linalg.jl")

# decompositions
include("decomposition.jl")

include("solve.jl")

# matrix copy
include("copy.jl")

end
