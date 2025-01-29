using Metal: metal_support
using Random
using SpecialFunctions

@testset "arguments" begin
    @on_device dispatch_quadgroups_per_threadgroup()
    @on_device dispatch_simdgroups_per_threadgroup()
    @on_device quadgroup_index_in_threadgroup()
    @on_device quadgroups_per_threadgroup()
    @on_device simdgroup_index_in_threadgroup()
    @on_device simdgroups_per_threadgroup()
    @on_device thread_index_in_quadgroup()
    @on_device thread_index_in_simdgroup()
    @on_device thread_index_in_threadgroup()
    @on_device thread_execution_width()
    @on_device threads_per_simdgroup()

    @on_device dispatch_threads_per_threadgroup_1d()
    @on_device dispatch_threads_per_threadgroup_2d()
    @on_device dispatch_threads_per_threadgroup_3d()

    @on_device grid_origin_1d()
    @on_device grid_origin_2d()
    @on_device grid_origin_3d()

    @on_device grid_size_1d()
    @on_device grid_size_2d()
    @on_device grid_size_3d()

    @on_device thread_position_in_grid_1d()
    @on_device thread_position_in_grid_2d()
    @on_device thread_position_in_grid_3d()

    @on_device thread_position_in_threadgroup_1d()
    @on_device thread_position_in_threadgroup_2d()
    @on_device thread_position_in_threadgroup_3d()

    @on_device threadgroup_position_in_grid_1d()
    @on_device threadgroup_position_in_grid_2d()
    @on_device threadgroup_position_in_grid_3d()

    @on_device threadgroups_per_grid_1d()
    @on_device threadgroups_per_grid_2d()
    @on_device threadgroups_per_grid_3d()

    @on_device threads_per_grid_1d()
    @on_device threads_per_grid_2d()
    @on_device threads_per_grid_3d()

    @on_device threads_per_threadgroup_1d()
    @on_device threads_per_threadgroup_2d()
    @on_device threads_per_threadgroup_3d()

    global const CPU_ONLY_ERR = "This function is not intended for use on the CPU"

    @test_throws CPU_ONLY_ERR dispatch_quadgroups_per_threadgroup()
    @test_throws CPU_ONLY_ERR dispatch_simdgroups_per_threadgroup()
    @test_throws CPU_ONLY_ERR quadgroup_index_in_threadgroup()
    @test_throws CPU_ONLY_ERR quadgroups_per_threadgroup()
    @test_throws CPU_ONLY_ERR simdgroup_index_in_threadgroup()
    @test_throws CPU_ONLY_ERR simdgroups_per_threadgroup()
    @test_throws CPU_ONLY_ERR thread_index_in_quadgroup()
    @test_throws CPU_ONLY_ERR thread_index_in_simdgroup()
    @test_throws CPU_ONLY_ERR thread_index_in_threadgroup()
    @test_throws CPU_ONLY_ERR thread_execution_width()
    @test_throws CPU_ONLY_ERR threads_per_simdgroup()

    @test_throws CPU_ONLY_ERR dispatch_threads_per_threadgroup_1d()
    @test_throws CPU_ONLY_ERR dispatch_threads_per_threadgroup_2d()
    @test_throws CPU_ONLY_ERR dispatch_threads_per_threadgroup_3d()

    @test_throws CPU_ONLY_ERR grid_origin_1d()
    @test_throws CPU_ONLY_ERR grid_origin_2d()
    @test_throws CPU_ONLY_ERR grid_origin_3d()

    @test_throws CPU_ONLY_ERR grid_size_1d()
    @test_throws CPU_ONLY_ERR grid_size_2d()
    @test_throws CPU_ONLY_ERR grid_size_3d()

    @test_throws CPU_ONLY_ERR thread_position_in_grid_1d()
    @test_throws CPU_ONLY_ERR thread_position_in_grid_2d()
    @test_throws CPU_ONLY_ERR thread_position_in_grid_3d()

    @test_throws CPU_ONLY_ERR thread_position_in_threadgroup_1d()
    @test_throws CPU_ONLY_ERR thread_position_in_threadgroup_2d()
    @test_throws CPU_ONLY_ERR thread_position_in_threadgroup_3d()

    @test_throws CPU_ONLY_ERR threadgroup_position_in_grid_1d()
    @test_throws CPU_ONLY_ERR threadgroup_position_in_grid_2d()
    @test_throws CPU_ONLY_ERR threadgroup_position_in_grid_3d()

    @test_throws CPU_ONLY_ERR threadgroups_per_grid_1d()
    @test_throws CPU_ONLY_ERR threadgroups_per_grid_2d()
    @test_throws CPU_ONLY_ERR threadgroups_per_grid_3d()

    @test_throws CPU_ONLY_ERR threads_per_grid_1d()
    @test_throws CPU_ONLY_ERR threads_per_grid_2d()
    @test_throws CPU_ONLY_ERR threads_per_grid_3d()

    @test_throws CPU_ONLY_ERR threads_per_threadgroup_1d()
    @test_throws CPU_ONLY_ERR threads_per_threadgroup_2d()
    @test_throws CPU_ONLY_ERR threads_per_threadgroup_3d()
end

############################################################################################

MATH_INTR_FUNCS_1_ARG = [
    # Common functions
    # saturate, # T saturate(T x) Clamp between 0.0 and 1.0
    sign, # T sign(T x) returns 0.0 if x is NaN

    # float math
    acos, # T acos(T x)
    asin, # T asin(T x)
    asinh, # T asinh(T x)
    atan, # T atan(T x)
    atanh, # T atanh(T x)
    ceil, # T ceil(T x)
    cos, # T cos(T x)
    cosh, # T cosh(T x)
    cospi, # T cospi(T x)
    exp, # T exp(T x)
    exp2, # T exp2(T x)
    exp10, # T exp10(T x)
    abs, #T [f]abs(T x)
    floor, # T floor(T x)
    Metal.fract, # T fract(T x)
    # ilogb, # Ti ilogb(T x)
    log, # T log(T x)
    log2, # T log2(T x)
    log10, # T log10(T x)
    # Metal.rint, # T rint(T x) # TODO: Add test. Not sure what the behaviour actually is
    round, # T round(T x)
    Metal.rsqrt, # T rsqrt(T x)
    sin, # T sin(T x)
    sinh, # T sinh(T x)
    sinpi, # T sinpi(T x)
    sqrt, # sqrt(T x)
    tan, # T tan(T x)
    tanh, # T tanh(T x)
    tanpi, # T tanpi(T x)
    trunc, # T trunc(T x)
]
Metal.rsqrt(x::Float16) = 1 / sqrt(x)
Metal.rsqrt(x::Float32) = 1 / sqrt(x)
Metal.fract(x::Float16) = mod(x, 1)
Metal.fract(x::Float32) = mod(x, 1)

MATH_INTR_FUNCS_2_ARG = [
    # Common function
    # step, # T step(T edge, T x) Returns 0.0 if x < edge, otherwise it returns 1.0

    # float math
    atan, # T atan2(T x, T y) Compute arc tangent of y over x.
    # fdim, # T fdim(T x, T y)
    max, # T [f]max(T x, T y)
    min, # T [f]min(T x, T y)
    # fmod, # T fmod(T x, T y)
    # frexp, # T frexp(T x, Ti &exponent)
    # ldexp, # T ldexp(T x, Ti k)
    # modf, # T modf(T x, T &intval)
    hypot, # NOT MSL but tested the same
]

MATH_INTR_FUNCS_3_ARG = [
    # Common functions
    # mix, # T mix(T x, T y, T a) # x+(y-x)*a
    # smoothstep, # T smoothstep(T edge0, T edge1, T x)
    fma, # T fma(T a, T b, T c)
    max, # T max3(T x, T y, T z)
    # median3, # T median3(T x, T y, T z)
    min, # T min3(T x, T y, T z)
]

@testset "math" begin
# 1-arg functions
@testset "$(fun)()::$T" for fun in MATH_INTR_FUNCS_1_ARG, T in (Float32, Float16)
    cpuarr = if fun in [log, log2, log10, Metal.rsqrt, sqrt]
        rand(T, 4)
    else
        T[0.0, -0.0, rand(T), -rand(T)]
    end

    mtlarr = MtlArray(cpuarr)

    mtlout = fill!(similar(mtlarr), 0)

    function kernel(res, arr)
        idx = thread_position_in_grid_1d()
        res[idx] = fun(arr[idx])
        return nothing
    end
    Metal.@sync @metal threads = length(mtlout) kernel(mtlout, mtlarr)
    @eval @test Array($mtlout) ≈ $fun.($cpuarr)
end
# 2-arg functions
@testset "$(fun)()::$T" for T in (Float32, Float16), fun in MATH_INTR_FUNCS_2_ARG
    N = 4
    arr1 = randn(T, N)
    arr2 = randn(T, N)
    mtlarr1 = MtlArray(arr1)
    mtlarr2 = MtlArray(arr2)

    mtlout = fill!(similar(mtlarr1), 0)

    function kernel(res, x, y)
        idx = thread_position_in_grid_1d()
        res[idx] = fun(x[idx], y[idx])
        return nothing
    end
    Metal.@sync @metal threads = N kernel(mtlout, mtlarr1, mtlarr2)
    @eval @test Array($mtlout) ≈ $fun.($arr1, $arr2)
end
# 3-arg functions
@testset "$(fun)()::$T" for T in (Float32, Float16), fun in MATH_INTR_FUNCS_3_ARG
    N = 4
    arr1 = randn(T, N)
    arr2 = randn(T, N)
    arr3 = randn(T, N)

    mtlarr1 = MtlArray(arr1)
    mtlarr2 = MtlArray(arr2)
    mtlarr3 = MtlArray(arr3)

    mtlout = fill!(similar(mtlarr1), 0)

    function kernel(res, x, y, z)
        idx = thread_position_in_grid_1d()
        res[idx] = fun(x[idx], y[idx], z[idx])
        return nothing
    end
    Metal.@sync @metal threads = N kernel(mtlout, mtlarr1, mtlarr2, mtlarr3)
    @eval @test Array($mtlout) ≈ $fun.($arr1, $arr2, $arr3)
end
end

@testset "unique math" begin
@testset "$T" for T in (Float32, Float16)
    let # acosh
        arr = T[0, rand(T, 3)...] .+ T(1)
        buffer = MtlArray(arr)
        vec = acosh.(buffer)
        @test Array(vec) ≈ acosh.(arr)
    end

    let # sincos
        N = 4
        arr = rand(T, N)
        bufferA = MtlArray(arr)
        bufferB = MtlArray(arr)
        function intr_test3(arr_sin, arr_cos)
            idx = thread_position_in_grid_1d()
            sinres, cosres = sincos(arr_cos[idx])
            arr_sin[idx] = sinres
            arr_cos[idx] = cosres
            return nothing
        end
        # Broken with Float16
        if T == Float16
            @test_broken Metal.@sync @metal threads = N intr_test3(bufferA, bufferB)
        else
            Metal.@sync @metal threads = N intr_test3(bufferA, bufferB)
            @test Array(bufferA) ≈ sin.(arr)
            @test Array(bufferB) ≈ cos.(arr)
        end
    end

    let # clamp
        N = 4
        in = randn(T, N)
        minval = fill(T(-0.6), N)
        maxval = fill(T(0.6), N)

        mtlin = MtlArray(in)
        mtlminval = MtlArray(minval)
        mtlmaxval = MtlArray(maxval)

        mtlout = fill!(similar(mtlin), 0)

        function kernel(res, x, y, z)
            idx = thread_position_in_grid_1d()
            res[idx] = clamp(x[idx], y[idx], z[idx])
            return nothing
        end
        Metal.@sync @metal threads = N kernel(mtlout, mtlin, mtlminval, mtlmaxval)
        @test Array(mtlout) == clamp.(in, minval, maxval)
    end

    let #pow
        N = 4
        arr1 = rand(T, N)
        arr2 = rand(T, N)
        mtlarr1 = MtlArray(arr1)
        mtlarr2 = MtlArray(arr2)

        mtlout = fill!(similar(mtlarr1), 0)

        function kernel(res, x, y)
            idx = thread_position_in_grid_1d()
            res[idx] = x[idx]^y[idx]
            return nothing
        end
        Metal.@sync @metal threads = N kernel(mtlout, mtlarr1, mtlarr2)
        @test Array(mtlout) ≈ arr1 .^ arr2
    end

    let #powr
        N = 4
        arr1 = rand(T, N)
        arr2 = rand(T, N)
        mtlarr1 = MtlArray(arr1)
        mtlarr2 = MtlArray(arr2)

        mtlout = fill!(similar(mtlarr1), 0)

        function kernel(res, x, y)
            idx = thread_position_in_grid_1d()
            res[idx] = Metal.powr(x[idx], y[idx])
            return nothing
        end
        Metal.@sync @metal threads = N kernel(mtlout, mtlarr1, mtlarr2)
        @test Array(mtlout) ≈ arr1 .^ arr2
    end

    let # log1p
        arr = collect(LinRange(nextfloat(-1.0f0), 10.0f0, 20))
        buffer = MtlArray(arr)
        vec = Array(log1p.(buffer))
        @test vec ≈ log1p.(arr)
    end

    let # erf
        arr = collect(LinRange(nextfloat(-3.0f0), 3.0f0, 20))
        buffer = MtlArray(arr)
        vec = Array(SpecialFunctions.erf.(buffer))
        @test vec ≈ SpecialFunctions.erf.(arr)
    end

    let # erfc
        arr = collect(LinRange(nextfloat(-3.0f0), 3.0f0, 20))
        buffer = MtlArray(arr)
        vec = Array(SpecialFunctions.erfc.(buffer))
        @test vec ≈ SpecialFunctions.erfc.(arr)
    end

    let # erfinv
        arr = collect(LinRange(-1.0f0, 1.0f0, 20))
        buffer = MtlArray(arr)
        vec = Array(SpecialFunctions.erfinv.(buffer))
        @test vec ≈ SpecialFunctions.erfinv.(arr)
    end

    let # expm1
        arr = collect(LinRange(nextfloat(-88.0f0), 88.0f0, 100))
        buffer = MtlArray(arr)
        vec = Array(expm1.(buffer))
        @test vec ≈ expm1.(arr)
    end


    let # nextafter
        if Metal.is_macos(v"14")
            N = 4
            function nextafter_test(X, y)
                idx = thread_position_in_grid_1d()
                X[idx] = Metal.nextafter(X[idx], y)
                return nothing
            end
            arr = rand(T, N)
            buffer = MtlArray(arr)
            Metal.@sync @metal threads = N nextafter_test(buffer, typemax(T))
            @test Array(buffer) == nextfloat.(arr)

            Metal.@sync @metal threads = N nextafter_test(buffer, typemin(T))
            @test Array(buffer) == arr
        end
    end
end
end

############################################################################################

@testset "complex" begin
    a = rand(ComplexF32,4)
    bufferA = MtlArray(a)
    vecA = Array(sqrt.(bufferA))
    @test vecA ≈ sqrt.(a)
end

############################################################################################

@testset "synchronization" begin
    # host/device synchronization
    let
        function sync_test_kernel(buf)
            idx = thread_position_in_grid_1d()
            @inbounds buf[idx] += 1
            return nothing
        end
        buf = Metal.zeros(Int, 1024; storage=Metal.SharedStorage)
        vec = unsafe_wrap(Vector{Int}, pointer(buf), size(buf))
        @metal threads=length(buf) sync_test_kernel(buf)
        synchronize()
        @test all(vec .== 1)
    end

    # thread synchronization
    let
        function barrier_test_kernel(buf)
            idx = thread_position_in_grid_1d()
            if thread_position_in_threadgroup_1d() != 1
                @inbounds buf[idx] = 1
            end

            threadgroup_barrier(Metal.MemoryFlagThreadGroup)

            if thread_position_in_threadgroup_1d() == 1
                for i in 2:threads_per_threadgroup_1d()
                    @inbounds buf[idx] += buf[i]
                end
            end
            return nothing
        end

        buf = Metal.zeros(Int, 1000)
        @metal threads=length(buf) barrier_test_kernel(buf)
        @test Array(buf)[1] == 999
    end

    # TODO: simdgroup barrier test
end

############################################################################################

@testset "memory" begin

# a composite type to test for more complex element types
@eval struct RGB{T}
    r::T
    g::T
    b::T
end

@testset "threadgroup memory" begin

n = 256

@testset "constructors" begin
    # static
    @on_device MtlThreadGroupArray(Float32, 1)
    @on_device MtlThreadGroupArray(Float32, (1,2))
    @on_device MtlThreadGroupArray(Tuple{Float32, Float32}, 1)
    @on_device MtlThreadGroupArray(Tuple{Float32, Float32}, (1,2))
    @on_device MtlThreadGroupArray(Tuple{RGB{Float32}, UInt32}, 1)
    @on_device MtlThreadGroupArray(Tuple{RGB{Float32}, UInt32}, (1,2))
end


@testset "static" begin

@testset "statically typed" begin
    function kernel(d, n)
        t = thread_position_in_threadgroup_1d()
        tr = n-t+1

        s = MtlThreadGroupArray(Float32, 1024)
        s2 = MtlThreadGroupArray(Float32, 1024)  # catch aliasing

        s[t] = d[t]
        s2[t] = 2*d[t]
        threadgroup_barrier()
        d[t] = s[tr]

        return
    end

    a = rand(Float32, n)
    d_a = MtlArray(a)

    @metal threads=n kernel(d_a, n)
    @test reverse(a) == Array(d_a)
end

@testset "parametrically typed" begin
    typs = [Int32, Int64, Float32]
    @testset for typ in typs
        function kernel(d::MtlDeviceArray{T}, n) where {T}
            t = thread_position_in_threadgroup_1d()
            tr = n-t+1

            s = MtlThreadGroupArray(T, 1024)
            s2 = MtlThreadGroupArray(T, 1024)  # catch aliasing

            s[t] = d[t]
            s2[t] = d[t]
            threadgroup_barrier()
            d[t] = s[tr]

            return
        end

        a = rand(typ, n)
        d_a = MtlArray(a)

        @metal threads=n kernel(d_a, n)
        @test reverse(a) == Array(d_a)
    end
end

end

end

end

############################################################################################

@testset "simd intrinsics" begin

@testset "shuffle($typ)" for typ in [Float32, Float16, Int32, UInt32, Int16, UInt16, Int8, UInt8]
    function kernel(a::MtlDeviceVector{T}, b::MtlDeviceVector{T}) where T
        idx = thread_position_in_grid_1d()
        idx_in_simd = thread_index_in_simdgroup()
        simd_idx = simdgroup_index_in_threadgroup()

        temp = MtlThreadGroupArray(T, 32)
        temp[idx] = a[idx]
        simdgroup_barrier(Metal.MemoryFlagThreadGroup)

        if simd_idx == 1
            value = temp[idx_in_simd]

            value = value + simd_shuffle_down(value, 16)
            value = value + simd_shuffle_down(value,  8)
            value = value + simd_shuffle_down(value,  4)
            value = value + simd_shuffle_down(value,  2)
            value = value + simd_shuffle_down(value,  1)

            b[idx] = value
        end
        return
    end

    dev_a = Metal.zeros(typ, 32; storage=Metal.SharedStorage)
    dev_b = Metal.zeros(typ, 32; storage=Metal.SharedStorage)
    a = unsafe_wrap(Array{typ}, dev_a, 32)
    b = unsafe_wrap(Array{typ}, dev_b, 32)

    rand!(a, (1:4))
    Metal.@sync @metal threads=32 kernel(dev_a, dev_b)
    @test sum(a) ≈ b[1]
end

@testset "matrix functions" begin
    @testset "load_store($typ)" for typ in [Float16, Float32]
        function kernel(a::MtlDeviceArray{T}, b::MtlDeviceArray{T},
                            origin_a=(1, 1), origin_b=(1, 1)) where {T}
            sg_a = simdgroup_load(a, origin_a)
            simdgroup_store(sg_a, b, origin_b)
            return
        end

        let
            a = MtlArray(rand(typ, 8, 8))
            b = MtlArray(zeros(typ, 8, 8))
            @metal threads=(8, 8) kernel(a, b)
            @test Array(a) == Array(b)
        end

        let
            a = MtlArray(rand(typ, 20, 15))
            b = MtlArray(zeros(typ, 15, 20))
            @metal threads=(8, 8) kernel(a, b, (4, 2), (3, 5))
            @test Array(a)[4:11, 2:9] == Array(b)[3:10, 5:12]
        end
    end

    @testset "load_store_tg($typ)" for typ in [Float16, Float32]
        function kernel(a::MtlDeviceArray{T}, b::MtlDeviceArray{T}) where {T}
            pos = thread_position_in_threadgroup_2d()

            tg_a = MtlThreadGroupArray(T, (8, 8))
            tg_a[pos.x, pos.y] = a[pos.x, pos.y]
            threadgroup_barrier(Metal.MemoryFlagThreadGroup)

            sg_a = simdgroup_load(tg_a)
            tg_b = MtlThreadGroupArray(T, (8, 8))
            simdgroup_store(sg_a, tg_b)

            threadgroup_barrier(Metal.MemoryFlagThreadGroup)
            b[pos.x, pos.y] = tg_b[pos.x, pos.y]

            return
        end

        a = MtlArray(rand(typ, 8, 8))
        b = MtlArray(zeros(typ, 8, 8))
        @metal threads=(8, 8) kernel(a, b)
        @test Array(a) == Array(b)
    end

    @testset "mul($typ)" for typ in [Float16, Float32]
        function kernel(a::MtlDeviceArray{T}, b::MtlDeviceArray{T}, c::MtlDeviceArray{T}) where {T}
            sg_a = simdgroup_load(a)
            sg_b = simdgroup_load(b)
            sg_c = simdgroup_multiply(sg_a, sg_b)
            simdgroup_store(sg_c, c)
            return
        end

        a = MtlArray(rand(typ, 8, 8))
        b = MtlArray(rand(typ, 8, 8))
        c = MtlArray(zeros(typ, 8, 8))
        @metal threads=(8, 8) kernel(a, b, c)
        @test Array(a) * Array(b) ≈ Array(c)
    end

    @testset "mad($typ)" for typ in [Float16, Float32]
        function kernel(a::MtlDeviceArray{T}, b::MtlDeviceArray{T}, c::MtlDeviceArray{T},
                    d::MtlDeviceArray{T}) where {T}
            sg_a = simdgroup_load(a)
            sg_b = simdgroup_load(b)
            sg_c = simdgroup_load(c)
            sg_d = simdgroup_multiply_accumulate(sg_a, sg_b, sg_c)
            simdgroup_store(sg_d, d)
            return
        end

        a = MtlArray(rand(typ, 8, 8))
        b = MtlArray(rand(typ, 8, 8))
        c = MtlArray(rand(typ, 8, 8))
        d = MtlArray(zeros(typ, 8, 8))
        @metal threads=(8, 8) kernel(a, b, c, d)
        @test Array(a) * Array(b) + Array(c) ≈ Array(d)
    end
end # End Matrix Functions

end # End SIMD Intrinsics


############################################################################################

@testset "atomics" begin

n = 128 # NOTE: also hard-coded in MtlThreadGroupArray constructors

# JuliaGPU/Metal.jl#217: threadgroup atomics seem to requires all-atomic operations

@testset "low-level" begin
    # TODO: make these tests actually write to the overlapping memory locations

    # XXX: according to the docs, Float32 atomics should also work on threadgroup memory

    @testset "store_explicit" begin
        function global_kernel(a, val)
            i = thread_position_in_grid_1d()
            Metal.atomic_store_explicit(pointer(a, i), val)
            return
        end

        types = [Int32]
        metal_support() >= v"3.0" && push!(types, Float32)
        @testset for T in types
            a = Metal.zeros(T, n)
            @metal threads=n global_kernel(a, T(42))
            @test all(isequal(42), Array(a))
        end

        function local_kernel(a, val::T) where T
            i = thread_position_in_grid_1d()
            b = MtlThreadGroupArray(T, 128)
            Metal.atomic_store_explicit(pointer(b, i), val)
            a[i] = b[i]
            return
        end

        @testset for T in [Int32,]
            a = Metal.zeros(T, n)
            @metal threads=n local_kernel(a, T(42))
            @test all(isequal(42), Array(a))
        end
    end

    @testset "load_explicit" begin
        function global_kernel(a, b)
            i = thread_position_in_grid_1d()
            val = Metal.atomic_load_explicit(pointer(a, i))
            b[i] = val
            return
        end

        types = [Int32]
        metal_support() >= v"3.0" && push!(types, Float32)
        @testset for T in types
            a = MtlArray(rand(T, n))
            b = Metal.zeros(T, n)
            @metal threads=n global_kernel(a, b)
            @test Array(a) == Array(b)
        end

        function local_kernel(a::AbstractArray{T}, b::AbstractArray{T}) where T
            i = thread_position_in_grid_1d()
            c = MtlThreadGroupArray(T, 128)
            #c[i] = a[i]
            val = Metal.atomic_load_explicit(pointer(a, i))
            Metal.atomic_store_explicit(pointer(c, i), val)
            val = Metal.atomic_load_explicit(pointer(c, i))
            #b[i] = val
            Metal.atomic_store_explicit(pointer(b, i), val)
            return
        end

        @testset for T in [Int32,]
            a = MtlArray(rand(T, n))
            b = Metal.zeros(T, n)
            @metal threads=n local_kernel(a, b)
            @test Array(a) == Array(b)
        end
    end

    @testset "exchange_explicit" begin
        function global_kernel(a, val)
            i = thread_position_in_grid_1d()
            Metal.atomic_exchange_explicit(pointer(a, i), val)
            return
        end

        types = [Int32]
        metal_support() >= v"3.0" && push!(types, Float32)
        @testset for T in types
            a = MtlArray(rand(T, n))
            @metal threads=n global_kernel(a, T(42))
            @test all(isequal(42), Array(a))
        end

        function local_kernel(a, val::T) where T
            i = thread_position_in_grid_1d()
            b = MtlThreadGroupArray(T, 128)
            Metal.atomic_exchange_explicit(pointer(b, i), val)
            a[i] = b[i]
            return
        end

        @testset for T in [Int32,]
            a = Metal.zeros(T, n)
            @metal threads=n local_kernel(a, T(42))
            @test all(isequal(42), Array(a))
        end
    end

    @testset "compare_exchange_weak_explicit" begin
        function global_kernel(a, expected, desired)
            i = thread_position_in_grid_1d()
            while Metal.atomic_compare_exchange_weak_explicit(pointer(a, i), expected[i], desired) != expected[i]
                # keep on trying
            end
            return
        end

        types = [Int32]
        metal_support() >= v"3.0" && push!(types, Float32)
        @testset for T in types
            a = MtlArray(rand(T, n))
            expected = copy(a)
            desired = T(42)
            @metal threads=length(a) global_kernel(a, expected, desired)
            @test all(isequal(42), Array(a))
        end

        function local_kernel(a, expected::AbstractArray{T}, desired::T) where T
            i = thread_position_in_grid_1d()
            b = MtlThreadGroupArray(T, 128)
            #b[i] = a[i]
            val = Metal.atomic_load_explicit(pointer(a, i))
            Metal.atomic_store_explicit(pointer(b, i), val)
            while Metal.atomic_compare_exchange_weak_explicit(pointer(b, i), expected[i], desired) != expected[i]
                # keep on trying
            end
            #a[i] = b[i]
            val = Metal.atomic_load_explicit(pointer(b, i))
            Metal.atomic_store_explicit(pointer(a, i), val)
            return
        end

        @testset for T in [Int32,]
            a = Metal.zeros(T, n)
            expected = copy(a)
            desired = T(42)
            @metal threads=n local_kernel(a, expected, desired)
            @test all(isequal(42), Array(a))
        end
    end

    @testset "fetch and modify" begin
        add_sub_types = [Int32, UInt32]
        metal_support() >= v"3.0" && push!(add_sub_types, Float32)
        other_types = [Int32, UInt32]
        for (jlfun, mtlfun, types) in [(min, Metal.atomic_fetch_min_explicit, other_types),
                                       (max, Metal.atomic_fetch_max_explicit, other_types),
                                       (&,   Metal.atomic_fetch_and_explicit, other_types),
                                       (|,   Metal.atomic_fetch_or_explicit,  other_types),
                                       (⊻,   Metal.atomic_fetch_xor_explicit, other_types),
                                       (+,   Metal.atomic_fetch_add_explicit, add_sub_types),
                                       (-,   Metal.atomic_fetch_sub_explicit, add_sub_types)
                                    ]
            function global_kernel(f, a, arg)
                i = thread_position_in_grid_1d()
                f(pointer(a, i), arg)
                return
            end

            function local_kernel(f, a, arg::T) where T
                i = thread_position_in_grid_1d()
                b = MtlThreadGroupArray(T, 128)
                #b[i] = a[i]
                val = Metal.atomic_load_explicit(pointer(a, i))
                Metal.atomic_store_explicit(pointer(b, i), val)
                f(pointer(b, i), arg)
                #a[i] = b[i]
                val = Metal.atomic_load_explicit(pointer(b, i))
                Metal.atomic_store_explicit(pointer(a, i), val)
                return
            end

            @testset "fetch_$(jlfun)_explicit" begin
                @testset "device $T" for T in types
                    a = rand(T, n)
                    b = MtlArray(a)
                    val = rand(T)
                    @metal threads=n global_kernel(mtlfun, b, val)
                    @test jlfun.(a, val) ≈ Array(b)
                end

                @testset "threadgroup $T" for T in setdiff(types, [Float32])
                    a = rand(T, n)
                    b = MtlArray(a)
                    val = rand(T)
                    @metal threads=n local_kernel(mtlfun, b, val)
                    @test jlfun.(a, val) ≈ Array(b)
                end
            end
        end
    end

    @testset "generic fetch and modify" begin
        # custom operator that doesn't map onto an atomic intrinsic
        f(a::T, b::T) where {T} = a + b + one(T)

        function global_kernel(a, op, arg)
            i = thread_position_in_grid_1d()
            Metal.atomic_fetch_op_explicit(pointer(a, i), op, arg)
            return
        end

        @testset for T in (Int32, UInt32)
            a = rand(T, n)
            b = MtlArray(a)
            val = rand(T)
            @metal threads=n global_kernel(b, f, val)
            @test f.(a, val) ≈ Array(b)
        end

        function local_kernel(a, op, arg::T) where T
            i = thread_position_in_grid_1d()
            b = MtlThreadGroupArray(T, 128)
            #b[i] = a[i]
            val = Metal.atomic_load_explicit(pointer(a, i))
            Metal.atomic_store_explicit(pointer(b, i), val)
            Metal.atomic_fetch_op_explicit(pointer(b, i), op, arg)
            #a[i] = b[i]
            val = Metal.atomic_load_explicit(pointer(b, i))
            Metal.atomic_store_explicit(pointer(a, i), val)
            return
        end

        @testset for T in (Int32, UInt32)
            a = rand(T, n)
            b = MtlArray(a)
            val = rand(T)
            @metal threads=n local_kernel(b, f, val)
            @test f.(a, val) ≈ Array(b)
        end
    end
end

@testset "high-level" begin
    # NOTE: this doesn't test threadgroup atomics, as those are assumed to have been
    #       covered by the low-level tests above, but only the atomic macro functionality.

    @testset "load" begin
        types = [Int32, UInt32]
        metal_support() >= v"3.0" && append!(types, [Float32])

        function kernel(a, b)
            i = thread_position_in_grid_1d()
            a[i] = Metal.@atomic b[i]
            return
        end

        @testset for T in types
            a = Metal.zeros(T, n)
            b = MtlArray(rand(T, n))
            @metal threads=n kernel(a, b)
            @test Array(a) == Array(b)
        end
    end

    @testset "store" begin
        types = [Int32, UInt32]
        metal_support() >= v"3.0" && append!(types, [Float32])

        function kernel(a, b)
            i = thread_position_in_grid_1d()
            val = b[i]
            Metal.@atomic a[i] = val
            return
        end

        @testset for T in types
            a = Metal.zeros(T, n)
            b = MtlArray(rand(T, n))
            @metal threads=n kernel(a, b)
            @test Array(a) == Array(b)
        end
    end

    @testset "add" begin
        types = [Int32, UInt32]
        metal_support() >= v"3.0" && append!(types, [Float32])

        function kernel(a)
            Metal.@atomic a[1] = a[1] + 1
            Metal.@atomic a[1] += 1
            return
        end

        @testset for T in types
            a = Metal.zeros(T)
            @metal threads=n kernel(a)
            @test Array(a)[1] == 2*n
        end
    end

    @testset "sub" begin
        types = [Int32, UInt32]
        metal_support() >= v"3.0" && append!(types, [Float32])

        function kernel(a)
            Metal.@atomic a[1] = a[1] - 1
            Metal.@atomic a[1] -= 1
            return
        end

        @testset for T in types
            a = MtlArray(T[2n])
            @metal threads=n kernel(a)
            @test Array(a)[1] == 0
        end
    end
end

end
