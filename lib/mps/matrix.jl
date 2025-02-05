## Some extra definitions for MPSDataType defined in libmps.jl

## bitwise operations lose type information, so allow conversions
Base.convert(::Type{MPSDataType}, x::Integer) = MPSDataType(x)

# Conversions for MPSDataTypes with Julia equivalents
const jl_mps_to_typ = Dict{MPSDataType, DataType}()
for type in [
        :Bool, :UInt8, :UInt16, :UInt32, :UInt64, :Int8, :Int16, :Int32, :Int64,
        :Float16, :BFloat16, :Float32, (:ComplexF16, :MPSDataTypeComplexFloat16),
        (:ComplexF32, :MPSDataTypeComplexFloat32),
    ]
    jltype, mpstype = if type isa Symbol
        type, Symbol(:MPSDataType, type)
    else
        type
    end
    @eval Base.convert(::Type{MPSDataType}, ::Type{$jltype}) = $(mpstype)
    @eval jl_mps_to_typ[$(mpstype)] = $jltype
end
Base.sizeof(t::MPSDataType) = sizeof(jl_mps_to_typ[t])

Base.convert(::Type{DataType}, mpstyp::MPSDataType) = jl_mps_to_typ[mpstyp]


## descriptor

export MPSMatrixDescriptor

# @objcwrapper MPSMatrixDescriptor <: NSObject

function MPSMatrixDescriptor(rows, columns, rowBytes, dataType)
    desc = @objc [MPSMatrixDescriptor matrixDescriptorWithRows:rows::NSUInteger
                                      columns:columns::NSUInteger
                                      rowBytes:rowBytes::NSUInteger
                                      dataType:dataType::MPSDataType]::id{MPSMatrixDescriptor}
    MPSMatrixDescriptor(desc)
end

function MPSMatrixDescriptor(rows, columns, matrices, rowBytes, matrixBytes, dataType)
    desc = @objc [MPSMatrixDescriptor matrixDescriptorWithRows:rows::NSUInteger
                                      columns:columns::NSUInteger
                                      matrices:matrices::NSUInteger
                                      rowBytes:rowBytes::NSUInteger
                                      matrixBytes:matrixBytes::NSUInteger
                                      dataType:dataType::MPSDataType]::id{MPSMatrixDescriptor}
    MPSMatrixDescriptor(desc)
end


## high-level object

export MPSMatrix

# @objcwrapper immutable=false MPSMatrix <: NSObject

function MPSMatrix(buf, descriptor::MPSMatrixDescriptor, offset::Integer=0)
    mat = @objc [MPSMatrix alloc]::id{MPSMatrix}
    obj = MPSMatrix(mat)
    finalizer(release, obj)
    @objc [obj::id{MPSMatrix} initWithBuffer:buf::id{MTLBuffer}
                              offset:offset::NSUInteger
                              descriptor:descriptor::id{MPSMatrixDescriptor}]::id{MPSMatrix}
    return obj
end

function MPSMatrix(dev::MTLDevice, descriptor::MPSMatrixDescriptor)
    mat = @objc [MPSMatrix alloc]::id{MPSMatrix}
    obj = MPSMatrix(mat)
    finalizer(release, obj)
    @objc [obj::id{MPSMatrix} initWithDevice:dev::id{MTLDevice}
                              descriptor:descriptor::id{MPSMatrixDescriptor}]::id{MPSMatrix}
    return obj
end

"""
    MPSMatrix(mat::MtlMatrix)

Metal matrix representation used in Performance Shaders.

Note that this results in a transposed view of the input,
as Metal stores matrices row-major instead of column-major.
"""
function MPSMatrix(mat::MtlMatrix{T}) where T
    n_cols, n_rows = size(mat)
    desc = MPSMatrixDescriptor(n_rows, n_cols, sizeof(T)*n_cols, T)
    offset = mat.offset * sizeof(T)
    return MPSMatrix(mat, desc, offset)
end

"""
    MPSMatrix(vec::MtlVector)

Metal matrix representation used in Performance Shaders.

Note that this results in a transposed view of the input,
as Metal stores matrices row-major instead of column-major.
"""
function MPSMatrix(vec::MtlVector{T}) where T
    n_cols, n_rows = length(vec), 1
    desc = MPSMatrixDescriptor(n_rows, n_cols, sizeof(T)*n_cols, T)
    offset = vec.offset * sizeof(T)
    return MPSMatrix(vec, desc, offset)
end

"""
    MPSMatrix(arr::MtlArray{T,3})

Metal batched matrix representation used in Performance Shaders.

Note that this results in a transposed view of the input,
as Metal stores matrices row-major instead of column-major.
"""
function MPSMatrix(arr::MtlArray{T,3}) where T
    n_cols, n_rows, n_matrices = size(arr)
    row_bytes = sizeof(T)*n_cols
    desc = MPSMatrixDescriptor(n_rows, n_cols, n_matrices, row_bytes, row_bytes * n_rows, T)
    offset = arr.offset * sizeof(T)
    return MPSMatrix(arr, desc, offset)
end


## matrix multiplication

export MPSMatrixMultiplication, encode!, matmul!

# @objcwrapper immutable=false MPSMatrixMultiplication <: MPSKernel

function MPSMatrixMultiplication(dev, transposeLeft, transposeRight, resultRows,
                                 resultColumns, interiorColumns, alpha, beta)
    kernel = @objc [MPSMatrixMultiplication alloc]::id{MPSMatrixMultiplication}
    obj = MPSMatrixMultiplication(kernel)
    finalizer(release, obj)
    @objc [obj::id{MPSMatrixMultiplication} initWithDevice:dev::id{MTLDevice}
                                            transposeLeft:transposeLeft::Bool
                                            transposeRight:transposeRight::Bool
                                            resultRows:resultRows::NSUInteger
                                            resultColumns:resultColumns::NSUInteger
                                            interiorColumns:interiorColumns::NSUInteger
                                            alpha:alpha::Cdouble
                                            beta:beta::Cdouble]::id{MPSMatrixMultiplication}
    return obj
end

function encode!(cmdbuf::MTLCommandBuffer, matmul::MPSMatrixMultiplication, left, right, result)
    @objc [matmul::id{MPSMatrixMultiplication} encodeToCommandBuffer:cmdbuf::id{MTLCommandBuffer}
                                               leftMatrix:left::id{MPSMatrix}
                                               rightMatrix:right::id{MPSMatrix}
                                               resultMatrix:result::id{MPSMatrix}]::Nothing
end

"""
    matmul!(a::MtlMatrix, b::MtlMatrix, c::MtlMatrix, alpha=1, beta=1,
              transpose_left=false, transpose_right=false)
A `MPSMatrixMultiplication` kernel that computes:
`c = alpha * op(a) * beta * op(b) + beta * C`

This function should not typically be used. Rather, use the normal `LinearAlgebra` interface
with any `MtlArray` and it should be accelerated using Metal Performance Shaders.
"""
function matmul!(c::MtlArray{T1,N}, a::MtlArray{T2,N}, b::MtlArray{T3,N},
                 alpha::Number=true, beta::Number=true,
    transpose_a=false, transpose_b=false) where {T1, T2, T3, N}
    # NOTE: MPS uses row major, while Julia is col-major. Instead of transposing
    #       the inputs (by passing !transpose_[ab]) and afterwards transposing
    #       the output, we use the property that (AB)ᵀ = BᵀAᵀ
    cols_a = size(a, transpose_a ? 1 : 2)
    cols_c, rows_c = size(c)

    # Create MPS-compatible matrix from the MtlArrays
    mps_a = MPSMatrix(a)
    mps_b = MPSMatrix(b)
    mps_c = MPSMatrix(c)

    mat_mul_kernel = MPSMatrixMultiplication(device(),
                                             transpose_b, transpose_a,
                                             rows_c, cols_c, cols_a,
                                             alpha, beta)


    # Encode and commit matmul kernel
    cmdbuf = MTLCommandBuffer(global_queue(device()))
    encode!(cmdbuf, mat_mul_kernel, mps_b, mps_a, mps_c)
    commit!(cmdbuf)

    return c
end


## topk

export MPSMatrixFindTopK, encode!

# @objcwrapper immutable=false MPSMatrixFindTopK <: MPSMatrixUnaryKernel

function MPSMatrixFindTopK(dev, numberOfTopKValues)
    kernel = @objc [MPSMatrixFindTopK alloc]::id{MPSMatrixFindTopK}
    obj = MPSMatrixFindTopK(kernel)
    finalizer(release, obj)
    @objc [obj::id{MPSMatrixFindTopK} initWithDevice:dev::id{MTLDevice}
                                                   numberOfTopKValues:numberOfTopKValues::NSUInteger]::id{MPSMatrixFindTopK}
    return obj
end

function encode!(cmdbuf::MTLCommandBuffer, kernel::MPSMatrixFindTopK, inputMatrix, resultIndexMatrix, resultValueMatrix)
    @objc [kernel::id{MPSMatrixFindTopK} encodeToCommandBuffer:cmdbuf::id{MTLCommandBuffer}
                                                      inputMatrix:inputMatrix::id{MPSMatrix}
                                                      resultIndexMatrix:resultIndexMatrix::id{MPSMatrix}
                                                      resultValueMatrix:resultValueMatrix::id{MPSMatrix}]::Nothing
end

"""
    MPS.topk!(A::MtlMatrix{T}, I::MtlMatrix{Int32}, V::MtlMatrix{T}, k)
                                                     where {T<:MtlFloat}

Compute the top `k` values and their corresponding indices column-wise in a matrix `A`.
Return the indices in `I` and the values in `V`.

`k` cannot be greater than 16.

Uses `MPSMatrixFindTopK`.

See also: [`topk`](@ref).

!!! warning
    This interface is experimental, and might change without warning.
"""
function topk!(A::MtlMatrix{T}, I::MtlMatrix{UInt32}, V::MtlMatrix{T}, k) where {T<:MtlFloat}
    size(I,1) >= k         || throw(ArgumentError("Matrix 'I' must be large enough for k rows"))
    size(I,2) >= size(A,2) || throw(ArgumentError("Matrix 'I' must have at least as many columns as A"))
    size(V,1) >= k         || throw(ArgumentError("Matrix 'V' must be large enough for k rows"))
    size(V,2) >= size(A,2) || throw(ArgumentError("Matrix 'V' must have at least as many columns as A"))

    return _topk!(A,I,V,k)
end
@inline function _topk!(A::MtlMatrix{T}, I::MtlMatrix{UInt32}, V::MtlMatrix{T}, k) where {T<:MtlFloat}
    size(A,1) >= k || throw(ArgumentError("Matrix 'A' must must have more rows than k"))
    k <= 16        || throw(ArgumentError("MPSMatrixFindTopK does not support values of k > 16"))

    # Create MPS-compatible matrix from the MtlArrays
    mps_a = MPSMatrix(A)
    mps_i = MPSMatrix(I)
    mps_v = MPSMatrix(V)

    topk_kernel = MPSMatrixFindTopK(device(), k)
    topk_kernel.indexOffset = 1

    # Encode and commit topk kernel
    cmdbuf = MTLCommandBuffer(global_queue(device()))
    encode!(cmdbuf, topk_kernel, mps_a, mps_i, mps_v)
    commit!(cmdbuf)

    return I, V
end

"""
    MPS.topk(A::MtlMatrix{T}, k) where {T<:MtlFloat}

Compute the top `k` values and their corresponding indices column-wise in a matrix `A`.
Return the indices in `I` and the values in `V`.

`k` cannot be greater than 16.

Uses `MPSMatrixFindTopK`.

See also: [`topk!`](@ref).

!!! warning
    This interface is experimental, and might change without warning.
"""
function topk(A::MtlMatrix{T,S}, k) where {T<:MtlFloat,S}
    s = (k,size(A,2))
    I = MtlMatrix{UInt32,S}(undef, s)
    V = MtlMatrix{T,S}(undef, s)

    return _topk!(A, I, V, k)
end


## softmax

export MPSMatrixSoftMax, MPSMatrixLogSoftMax, encode!

# @objcwrapper immutable=false MPSMatrixSoftMax <: MPSMatrixUnaryKernel
# @objcwrapper immutable=false MPSMatrixLogSoftMax <: MPSMatrixSoftMax

for f in (:MPSMatrixSoftMax, :MPSMatrixLogSoftMax)
    @eval begin
        function $(f)(dev)
            kernel = @objc [$(f) alloc]::id{$(f)}
            obj = $(f)(kernel)
            finalizer(release, obj)
            @objc [obj::id{$(f)} initWithDevice:dev::id{MTLDevice}]::id{$(f)}
            return obj
        end

        function encode!(cmdbuf::MTLCommandBuffer, kernel::$(f), inputMatrix, resultMatrix)
            @objc [kernel::id{$(f)} encodeToCommandBuffer:cmdbuf::id{MTLCommandBuffer}
                                    inputMatrix:inputMatrix::id{MPSMatrix}
                                    resultMatrix:resultMatrix::id{MPSMatrix}]::Nothing
        end
    end
end
