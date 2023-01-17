export simdgroup_load, simdgroup_store, simdgroup_multiply, simdgroup_multiply_accumulate;

using Core: LLVMPtr

function convert_origin(origin::NTuple{2, Int64})
    return (VecElement{Int64}(origin[1]-1), VecElement{Int64}(origin[2]-1))
end

for (jltype, suffix) in ((Float16, "f16"), (Float32, "f32"))
    for as in (AS.Device, AS.ThreadGroup)
        @eval begin
            @device_function simdgroup_load(
                data::Union{MtlDeviceArray{$jltype, <:Any, $as}, MtlLargerDeviceArray{$jltype, <:Any, $as}},
                matrix_origin::NTuple{2, Int64} = (1, 1),
            ) = @typed_ccall($"air.simdgroup_matrix_8x8_load.v64$suffix.p$as$suffix",
                llvmcall, NTuple{64, VecElement{$jltype}},
                (LLVMPtr{$jltype, $as}, Int64, NTuple{2, VecElement{Int64}}, Bool),
                pointer(data), size(data)[1], convert_origin(matrix_origin), Val(true))

            @device_function simdgroup_store(
                src::NTuple{64, VecElement{$jltype}},
                dest::Union{MtlDeviceArray{$jltype, <:Any, $as}, MtlLargerDeviceArray{$jltype, <:Any, $as}},
                matrix_origin::NTuple{2, Int64} = (1, 1),
            ) = @typed_ccall($"air.simdgroup_matrix_8x8_store.v64$suffix.p$as$suffix",
                llvmcall, Cvoid,
                (NTuple{64, VecElement{$jltype}}, LLVMPtr{$jltype, $as}, Int64, NTuple{2, VecElement{Int64}}, Bool),
                src, pointer(dest), size(dest)[1], convert_origin(matrix_origin), Val(true))
        end
    end

    @eval begin
        @device_function simdgroup_multiply(
            a::NTuple{64, VecElement{$jltype}},
            b::NTuple{64, VecElement{$jltype}},
        ) = ccall($"extern air.simdgroup_matrix_8x8_multiply_accumulate.v64$suffix.v64$suffix.v64$suffix.v64$suffix",
            llvmcall, NTuple{64, VecElement{$jltype}},
            (NTuple{64, VecElement{$jltype}}, NTuple{64, VecElement{$jltype}}, NTuple{64, VecElement{$jltype}}),
            a, b, ntuple(_ -> VecElement{$jltype}(0.0), Val(64)))

        @device_function simdgroup_multiply_accumulate(
            a::NTuple{64, VecElement{$jltype}},
            b::NTuple{64, VecElement{$jltype}},
            c::NTuple{64, VecElement{$jltype}},
        ) = ccall($"extern air.simdgroup_matrix_8x8_multiply_accumulate.v64$suffix.v64$suffix.v64$suffix.v64$suffix",
            llvmcall, NTuple{64, VecElement{$jltype}},
            (NTuple{64, VecElement{$jltype}}, NTuple{64, VecElement{$jltype}}, NTuple{64, VecElement{$jltype}}),
            a, b, c)
    end
end

## Documentation

@doc """
    simdgroup_load(data::MtlDeviceArray{T}, matrix_origin=(1, 1))

Loads data from device or threadgroup memory into an 8x8 SIMD-group matrix
and returns it. `T` must be either `Float16` or `Float32`.

# Arguments
- `matrix_origin::NTuple{2, Int64}=(1, 1)`: origin in the source memory to load from.
""" simdgroup_load

@doc """
    simdgroup_store(src, dest::MtlDeviceArray{T}, matrix_origin=(1, 1))

Stores data from an 8x8 SIMD-group matrix into device or threadgroup memory.
`T` must be either `Float16` or `Float32`.

# Arguments
- `matrix_origin::NTuple{2, Int64}=(1, 1)`: origin in the destination memory to store to.
""" simdgroup_store

@doc """
    simdgroup_multiply(a, b)

Returns `a * b`.
""" simdgroup_multiply

@doc """
    simdgroup_multiply_accumulate(a, b, c)

Returns `a * b + c`.
""" simdgroup_multiply_accumulate
