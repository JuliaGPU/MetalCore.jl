# Math function mappings to Metal intrinsics

using Base: FastMath
using Base.Math: throw_complex_domainerror

# TODO:
# - wrap all intrinsics from include/metal/metal_math
# - add support for vector types
# - consider emitting LLVM intrinsics and lowering those in the back-end

### Common Intrinsics
@device_function clamp_fast(x::Float32, minval::Float32, maxval::Float32) = ccall("extern air.fast_clamp.f32", llvmcall, Cfloat, (Cfloat, Cfloat, Cfloat), x, minval, maxval)
@device_override Base.clamp(x::Float32, minval::Float32, maxval::Float32) = ccall("extern air.clamp.f32", llvmcall, Cfloat, (Cfloat, Cfloat, Cfloat), x, minval, maxval)
@device_override Base.clamp(x::Float16, minval::Float16, maxval::Float16) = ccall("extern air.clamp.f16", llvmcall, Float16, (Float16, Float16, Float16), x, minval, maxval)

@device_override Base.sign(x::Float32) = ccall("extern air.sign.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.sign(x::Float16) = ccall("extern air.sign.f16", llvmcall, Float16, (Float16,), x)

### Floating Point Intrinsics

## Metal only supports single and half-precision floating-point types (and their vector counterparts)
## For single precision types, there are precise and fast variants

@device_override FastMath.abs_fast(x::Float32) = ccall("extern air.fast_fabs.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.abs(x::Float32) = ccall("extern air.fabs.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.abs(x::Float16) = ccall("extern air.fabs.f16", llvmcall, Float16, (Float16,), x)

@device_override FastMath.min_fast(x::Float32, y::Float32) = ccall("extern air.fast_fmin.f32", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
@device_override Base.min(x::Float32, y::Float32) = ccall("extern air.fmin.f32", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
@device_override Base.min(x::Float16, y::Float16) = ccall("extern air.fmin.f16", llvmcall, Float16, (Float16, Float16), x, y)

@device_override FastMath.min_fast(x::Float32, y::Float32, z::Float32) = ccall("extern air.fast_fmin3.f32", llvmcall, Cfloat, (Cfloat, Cfloat, Cfloat), x, y, z)
@device_override Base.min(x::Float32, y::Float32, z::Float32) = ccall("extern air.fmin3.f32", llvmcall, Cfloat, (Cfloat, Cfloat, Cfloat), x, y, z)
@device_override Base.min(x::Float16, y::Float16, z::Float16) = ccall("extern air.fmin3.f16", llvmcall, Float16, (Float16, Float16, Float16), x, y, z)

@device_override FastMath.max_fast(x::Float32, y::Float32) = ccall("extern air.fast_fmax.f32", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
@device_override Base.max(x::Float32, y::Float32) = ccall("extern air.fmax.f32", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
@device_override Base.max(x::Float16, y::Float16) = ccall("extern air.fmax.f16", llvmcall, Float16, (Float16, Float16), x, y)

@device_override FastMath.max_fast(x::Float32, y::Float32, z::Float32) = ccall("extern air.fast_fmax3.f32", llvmcall, Cfloat, (Cfloat, Cfloat, Cfloat), x, y, z)
@device_override Base.max(x::Float32, y::Float32, z::Float32) = ccall("extern air.fmax3.f32", llvmcall, Cfloat, (Cfloat, Cfloat, Cfloat), x, y, z)
@device_override Base.max(x::Float16, y::Float16, z::Float16) = ccall("extern air.fmax3.f16", llvmcall, Float16, (Float16, Float16, Float16), x, y, z)

@device_override FastMath.acos_fast(x::Float32) = ccall("extern air.fast_acos.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.acos(x::Float32) = ccall("extern air.acos.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.acos(x::Float16) = ccall("extern air.acos.f16", llvmcall, Float16, (Float16,), x)

@device_override FastMath.acosh_fast(x::Float32) = ccall("extern air.fast_acosh.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.acosh(x::Float32) = ccall("extern air.acosh.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.acosh(x::Float16) = ccall("extern air.acosh.f16", llvmcall, Float16, (Float16,), x)

@device_override FastMath.asin_fast(x::Float32) = ccall("extern air.fast_asin.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.asin(x::Float32) = ccall("extern air.asin.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.asin(x::Float16) = ccall("extern air.asin.f16", llvmcall, Float16, (Float16,), x)

@device_override FastMath.asinh_fast(x::Float32) = ccall("extern air.fast_asinh.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.asinh(x::Float32) = ccall("extern air.asinh.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.asinh(x::Float16) = ccall("extern air.asinh.f16", llvmcall, Float16, (Float16,), x)

@device_override FastMath.atan_fast(x::Float32) = ccall("extern air.fast_atan.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.atan(x::Float32) = ccall("extern air.atan.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.atan(x::Float16) = ccall("extern air.atan.f16", llvmcall, Float16, (Float16,), x)

@device_override FastMath.atan_fast(x::Float32, y::Float32) = ccall("extern air.fast_atan2.f32", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
@device_override Base.atan(x::Float32, y::Float32) = ccall("extern air.atan2.f32", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
@device_override Base.atan(x::Float16, y::Float16) = ccall("extern air.atan2.f16", llvmcall, Float16, (Float16, Float16), x, y)

@device_override FastMath.atanh_fast(x::Float32) = ccall("extern air.fast_atanh.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.atanh(x::Float32) = ccall("extern air.atanh.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.atanh(x::Float16) = ccall("extern air.atanh.f16", llvmcall, Float16, (Float16,), x)

@device_function ceil_fast(x::Float32) = ccall("extern air.fast_ceil.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.ceil(x::Float32) = ccall("extern air.ceil.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.ceil(x::Float16) = ccall("extern air.ceil.f16", llvmcall, Float16, (Float16,), x)

@device_override FastMath.cos_fast(x::Float32) = ccall("extern air.fast_cos.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.cos(x::Float32) = ccall("extern air.cos.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.cos(x::Float16) = ccall("extern air.cos.f16", llvmcall, Float16, (Float16,), x)

@device_override FastMath.cosh_fast(x::Float32) = ccall("extern air.fast_cosh.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.cosh(x::Float32) = ccall("extern air.cosh.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.cosh(x::Float16) = ccall("extern air.cosh.f16", llvmcall, Float16, (Float16,), x)

@device_function cospi_fast(x::Float32) = ccall("extern air.fast_cospi.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.cospi(x::Float32) = ccall("extern air.cospi.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.cospi(x::Float16) = ccall("extern air.cospi.f16", llvmcall, Float16, (Float16,), x)

@device_override FastMath.exp_fast(x::Float32) = ccall("extern air.fast_exp.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.exp(x::Float32) = ccall("extern air.exp.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.exp(x::Float16) = ccall("extern air.exp.f16", llvmcall, Float16, (Float16,), x)

@device_override FastMath.exp2_fast(x::Float32) = ccall("extern air.fast_exp2.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.exp2(x::Float32) = ccall("extern air.exp2.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.exp2(x::Float16) = ccall("extern air.exp2.f16", llvmcall, Float16, (Float16,), x)

@device_override FastMath.exp10_fast(x::Float32) = ccall("extern air.fast_exp10.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.exp10(x::Float32) = ccall("extern air.exp10.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.exp10(x::Float16) = ccall("extern air.exp10.f16", llvmcall, Float16, (Float16,), x)

@device_function floor_fast(x::Float32) = ccall("extern air.fast_floor.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.floor(x::Float32) = ccall("extern air.floor.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.floor(x::Float16) = ccall("extern air.floor.f16", llvmcall, Float16, (Float16,), x)

# half/regular -> air.fma.f16
# half/(precise or fast) -> air.fma.f32
@device_override Base.fma(a::Float32, b::Float32, c::Float32) = ccall("extern air.fma.f32", llvmcall, Float32, (Float32,Float32,Float32,), a,b,c)
@device_override Base.fma(a::Float16, b::Float16, c::Float16) = ccall("extern air.fma.f32", llvmcall, Float16, (Float16,Float16,Float16,), a,b,c)

@device_function fract_fast(x::Float32) = ccall("extern air.fast_fract.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_function fract(x::Float32) = ccall("extern air.fract.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_function fract(x::Float16) = ccall("extern air.fract.f16", llvmcall, Float16, (Float16,), x)

@device_override FastMath.log_fast(x::Float32) = ccall("extern air.fast_log.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.log(x::Float32) = ccall("extern air.log.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.log(x::Float16) = ccall("extern air.log.f16", llvmcall, Float16, (Float16,), x)

@device_override FastMath.log2_fast(x::Float32) = ccall("extern air.fast_log2.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.log2(x::Float32) = ccall("extern air.log2.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.log2(x::Float16) = ccall("extern air.log2.f16", llvmcall, Float16, (Float16,), x)

@device_override FastMath.log10_fast(x::Float32) = ccall("extern air.fast_log10.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.log10(x::Float32) = ccall("extern air.log10.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.log10(x::Float16) = ccall("extern air.log10.f16", llvmcall, Float16, (Float16,), x)

# Implementation of `log1p(::Float32)` from openlibm's `log1pf`
# https://github.com/JuliaMath/openlibm
const ln2_hi = 0.6931381f0
const ln2_lo = 9.058001f-6
const Lp1 = 0.6666667f0
const Lp2 = 0.4f0
const Lp3 = 0.2857143f0
const Lp4 = 0.22222199f0
const Lp5 = 0.18183573f0
const Lp6 = 0.15313838f0
const Lp7 = 0.14798199f0

@device_override function Base.Math.log1p(x::Float32)
    hx = reinterpret(Int32, x)
    ax = hx & 0x7fffffff  # |x|

    k = 1
    if hx < 0x3ed413d0  # x < sqrt(2) - 1
        if ax >= 0x3f800000  # |x| ≥ 1
            if x == -1
                return -Inf32
            elseif isnan(x)
                return NaN32
            else  # x < -1
                # TODO: switch to throw_complex_domainerror_neg1 for next Julia release
                throw_complex_domainerror(:log1p, x)
            end
        end

        if ax < 0x38000000  # |x| < 2^-15
            if ax < 0x33800000 # |x| < 2^-24
                return x  # Inexact
            else
                return x - x * x * 0.5f0
            end
        end

        if hx > 0 || hx <= reinterpret(Int32, 0xbe95f619)  # (sqrt(2)/2)-1 <= x
            k = 0
            f = x
            hu = 1f0
        end
    end  # hx < 0x3ed413d0

    if hx >= 0x7f800000
        return x + x
    end

    if k ≠ 0
        if hx < 0x5a000000
            u = 1f0 + x
            hu = reinterpret(Int32, u)
            k = (hu >> 23) - 127
            c = k > 0 ? 1f0 - (u - x) : x - (u - 1f0)
            c /= u
        else
            u = x
            hu = reinterpret(Int32, u)
            k = (hu >> 23) - 127
            c = 0f0
        end

        hu &= 0x007fffff

        if hu < 0x3504f4  # u < sqrt(2)
            u = reinterpret(Float32, hu | 0x3f800000)
        else
            k += 1
            u = reinterpret(Float32, hu | 0x3f000000)
            hu = (0x00800000 - hu) >> 2
        end
        f = u - 1f0
    end

    hfsq = 0.5f0 * f * f

    if hu == 0  # |f| < 2^-20
        if f == 0
            if k == 0
                return 0f0
            else
                c += k * ln2_lo
                return k * ln2_hi + c
            end
        end
        R = hfsq * (1f0 - Lp1 * f)
        if k == 0
            return f - R
        else
            return k * ln2_hi - ((R - (k * ln2_lo + c)) - f)
        end
    end

    s = f / (2f0 + f)
    z = s * s
    R = z * (Lp1 + z * (Lp2 + z * (Lp3 + z * (Lp4 + z * (Lp5 + z * (Lp6 + z * Lp7))))))
    if k == 0
        return f - (hfsq - s * (hfsq + R))
    else
        return k * ln2_hi - ((hfsq - (s * (hfsq + R) + (k * ln2_lo + c))) - f)
    end
end


@device_override FastMath.pow_fast(x::Float32, y::Float32) = ccall("extern air.fast_pow.f32", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
@device_override Base.:(^)(x::Float32, y::Float32) = ccall("extern air.pow.f32", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
@device_override Base.:(^)(x::Float16, y::Float16) = ccall("extern air.pow.f16", llvmcall, Float16, (Float16, Float16), x, y)

@device_function powr_fast(x::Float32, y::Float32) = ccall("extern air.fast_powr.f32", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
@device_function powr(x::Float32, y::Float32) = ccall("extern air.powr.f32", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
@device_function powr(x::Float16, y::Float16) = ccall("extern air.powr.f16", llvmcall, Float16, (Float16, Float16), x, y)

@device_function rint_fast(x::Float32) = ccall("extern air.fast_rint.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_function rint(x::Float32) = ccall("extern air.rint.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_function rint(x::Float16) = ccall("extern air.rint.f16", llvmcall, Float16, (Float16,), x)

@device_function round_fast(x::Float32) = ccall("extern air.fast_round.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.round(x::Float32) = ccall("extern air.round.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.round(x::Float16) = ccall("extern air.round.f16", llvmcall, Float16, (Float16,), x)

@device_function rsqrt_fast(x::Float32) = ccall("extern air.fast_rsqrt.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_function rsqrt(x::Float32) = ccall("extern air.rsqrt.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_function rsqrt(x::Float16) = ccall("extern air.rsqrt.f16", llvmcall, Float16, (Float16,), x)

@device_override FastMath.sin_fast(x::Float32) = ccall("extern air.fast_sin.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.sin(x::Float32) = ccall("extern air.sin.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.sin(x::Float16) = ccall("extern air.sin.f16", llvmcall, Float16, (Float16,), x)

@device_override function FastMath.sincos_fast(x::Float32)
    c = Ref{Cfloat}()
    s = ccall("extern air.fast_sincos.f32", llvmcall, Cfloat, (Cfloat, Ptr{Cfloat}), x, c)
    (s, c[])
end
@device_override function Base.sincos(x::Float32)
    c = Ref{Cfloat}()
    s = ccall("extern air.sincos.f32", llvmcall, Cfloat, (Cfloat, Ptr{Cfloat}), x, c)
    (s, c[])
end
# XXX: Broken
@device_override function Base.sincos(x::Float16)
    c = Ref{Float16}()
    s = ccall("extern air.sincos.f16", llvmcall, Float16, (Float16, Ptr{Float16}), x, c)
    (s, c[])
end

@device_override FastMath.sinh_fast(x::Float32) = ccall("extern air.fast_sinh.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.sinh(x::Float32) = ccall("extern air.sinh.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.sinh(x::Float16) = ccall("extern air.sinh.f16", llvmcall, Float16, (Float16,), x)

@device_function sinpi_fast(x::Float32) = ccall("extern air.fast_sinpi.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.sinpi(x::Float32) = ccall("extern air.sinpi.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.sinpi(x::Float16) = ccall("extern air.sinpi.f16", llvmcall, Float16, (Float16,), x)

@device_override FastMath.sqrt_fast(x::Float32) = ccall("extern air.fast_sqrt.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.sqrt(x::Float32) = ccall("extern air.sqrt.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.sqrt(x::Float16) = ccall("extern air.sqrt.f16", llvmcall, Float16, (Float16,), x)

@device_override FastMath.tan_fast(x::Float32) = ccall("extern air.fast_tan.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.tan(x::Float32) = ccall("extern air.tan.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.tan(x::Float16) = ccall("extern air.tan.f16", llvmcall, Float16, (Float16,), x)

@device_override FastMath.tanh_fast(x::Float32) = ccall("extern air.fast_tanh.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.tanh(x::Float32) = ccall("extern air.tanh.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.tanh(x::Float16) = ccall("extern air.tanh.f16", llvmcall, Float16, (Float16,), x)

@device_function tanpi_fast(x::Float32) = ccall("extern air.fast_tanpi.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.tanpi(x::Float32) = ccall("extern air.tanpi.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.tanpi(x::Float16) = ccall("extern air.tanpi.f16", llvmcall, Float16, (Float16,), x)

@device_function trunc_fast(x::Float32) = ccall("extern air.fast_trunc.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.trunc(x::Float32) = ccall("extern air.trunc.f32", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.trunc(x::Float16) = ccall("extern air.trunc.f16", llvmcall, Float16, (Float16,), x)

@static if Metal.is_macos(v"14")
    @device_function nextafter(x::Float32, y::Float32) = ccall("extern air.nextafter.f32", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
    @device_function nextafter(x::Float16, y::Float16) = ccall("extern air.nextafter.f16", llvmcall, Float16, (Float16, Float16), x, y)
end

# hypot without use of double
#
# taken from Cosmopolitan Libc
# Copyright 2021 Justine Alexandra Roberts Tunney
@inline function _hypot(a::T, b::T) where T <: AbstractFloat
    if isinf(a) || isinf(b)
        return T(Inf)
    end
    a = abs(a)
    b = abs(b)
    if a < b
        b, a = a, b
    end
    if iszero(a)
        return b
    end
    r = b / a
    return a * sqrt(one(T) + r * r)
end
@device_override Base.hypot(x::Float32, y::Float32) = _hypot(x, y)
@device_override Base.hypot(x::Float16, y::Float16) = _hypot(x, y)


### Integer Intrinsics

@device_override Base.abs(x::Int64)   = ccall("extern air.abs.s.i64", llvmcall, Int64, (Int64,), x)
@device_override Base.abs(x::UInt64)  = ccall("extern air.abs.u.i64", llvmcall, UInt64, (UInt64,), x)
@device_override Base.abs(x::Int32)   = ccall("extern air.abs.s.i32", llvmcall, Int32, (Int32,), x)
@device_override Base.abs(x::UInt32)  = ccall("extern air.abs.u.i32", llvmcall, UInt32, (UInt32,), x)
@device_override Base.abs(x::Int16)   = ccall("extern air.abs.s.i16", llvmcall, Int16, (Int16,), x)
@device_override Base.abs(x::UInt16)  = ccall("extern air.abs.u.i16", llvmcall, UInt16, (UInt16,), x)
@device_override Base.abs(x::Int8)    = ccall("extern air.abs.s.i8", llvmcall, Int8, (Int8,), x)
@device_override Base.abs(x::UInt8)   = ccall("extern air.abs.u.i8", llvmcall, UInt8, (UInt8,), x)

@device_override Base.min(x::Int64)   = ccall("extern air.min.s.i64", llvmcall, Int64, (Int64,), x)
@device_override Base.min(x::UInt64)  = ccall("extern air.min.u.i64", llvmcall, UInt64, (UInt64,), x)
@device_override Base.min(x::Int32)   = ccall("extern air.min.s.i32", llvmcall, Int32, (Int32,), x)
@device_override Base.min(x::UInt32)  = ccall("extern air.min.u.i32", llvmcall, UInt32, (UInt32,), x)
@device_override Base.min(x::Int16)   = ccall("extern air.min.s.i16", llvmcall, Int16, (Int16,), x)
@device_override Base.min(x::UInt16)  = ccall("extern air.min.u.i16", llvmcall, UInt16, (UInt16,), x)
@device_override Base.min(x::Int8)    = ccall("extern air.min.s.i8", llvmcall, Int8, (Int8,), x)
@device_override Base.min(x::UInt8)   = ccall("extern air.min.u.i8", llvmcall, UInt8, (UInt8,), x)

@device_override Base.max(x::Int64)   = ccall("extern air.max.s.i64", llvmcall, Int64, (Int64,), x)
@device_override Base.max(x::UInt64)  = ccall("extern air.max.u.i64", llvmcall, UInt64, (UInt64,), x)
@device_override Base.max(x::Int32)   = ccall("extern air.max.s.i32", llvmcall, Int32, (Int32,), x)
@device_override Base.max(x::UInt32)  = ccall("extern air.max.u.i32", llvmcall, UInt32, (UInt32,), x)
@device_override Base.max(x::Int16)   = ccall("extern air.max.s.i16", llvmcall, Int16, (Int16,), x)
@device_override Base.max(x::UInt16)  = ccall("extern air.max.u.i16", llvmcall, UInt16, (UInt16,), x)
@device_override Base.max(x::Int8)    = ccall("extern air.max.s.i8", llvmcall, Int8, (Int8,), x)
@device_override Base.max(x::UInt8)   = ccall("extern air.max.u.i8", llvmcall, UInt8, (UInt8,), x)

@device_function clz(x::Int64)   = ccall("extern air.clz.i64", llvmcall, Int64, (Int64,), x)
@device_function clz(x::UInt64)  = ccall("extern air.clz.i64", llvmcall, UInt64, (UInt64,), x)
@device_function clz(x::Int32)   = ccall("extern air.clz.i32", llvmcall, Int32, (Int32,), x)
@device_function clz(x::UInt32)  = ccall("extern air.clz.i32", llvmcall, UInt32, (UInt32,), x)
@device_function clz(x::Int16)   = ccall("extern air.clz.i16", llvmcall, Int16, (Int16,), x)
@device_function clz(x::UInt16)  = ccall("extern air.clz.i16", llvmcall, UInt16, (UInt16,), x)
@device_function clz(x::Int8)    = ccall("extern air.clz.i8", llvmcall, Int8, (Int8,), x)
@device_function clz(x::UInt8)   = ccall("extern air.clz.i8", llvmcall, UInt8, (UInt8,), x)

@device_function ctz(x::Int64)   = ccall("extern air.ctz.i64", llvmcall, Int64, (Int64,), x)
@device_function ctz(x::UInt64)  = ccall("extern air.ctz.i64", llvmcall, UInt64, (UInt64,), x)
@device_function ctz(x::Int32)   = ccall("extern air.ctz.i32", llvmcall, Int32, (Int32,), x)
@device_function ctz(x::UInt32)  = ccall("extern air.ctz.i32", llvmcall, UInt32, (UInt32,), x)
@device_function ctz(x::Int16)   = ccall("extern air.ctz.i16", llvmcall, Int16, (Int16,), x)
@device_function ctz(x::UInt16)  = ccall("extern air.ctz.i16", llvmcall, UInt16, (UInt16,), x)
@device_function ctz(x::Int8)    = ccall("extern air.ctz.i8", llvmcall, Int8, (Int8,), x)
@device_function ctz(x::UInt8)   = ccall("extern air.ctz.i8", llvmcall, UInt8, (UInt8,), x)

@device_function popcount(x::Int64)   = ccall("extern air.popcount.i64", llvmcall, Int64, (Int64,), x)
@device_function popcount(x::UInt64)  = ccall("extern air.popcount.i64", llvmcall, UInt64, (UInt64,), x)
@device_function popcount(x::Int32)   = ccall("extern air.popcount.i32", llvmcall, Int32, (Int32,), x)
@device_function popcount(x::UInt32)  = ccall("extern air.popcount.i32", llvmcall, UInt32, (UInt32,), x)
@device_function popcount(x::Int16)   = ccall("extern air.popcount.i16", llvmcall, Int16, (Int16,), x)
@device_function popcount(x::UInt16)  = ccall("extern air.popcount.i16", llvmcall, UInt16, (UInt16,), x)
@device_function popcount(x::Int8)    = ccall("extern air.popcount.i8", llvmcall, Int8, (Int8,), x)
@device_function popcount(x::UInt8)   = ccall("extern air.popcount.i8", llvmcall, UInt8, (UInt8,), x)

@device_function reverse_bits(x::Int64)   = ccall("extern air.reverse_bits.i64", llvmcall, Int64, (Int64,), x)
@device_function reverse_bits(x::UInt64)  = ccall("extern air.reverse_bits.i64", llvmcall, UInt64, (UInt64,), x)
@device_function reverse_bits(x::Int32)   = ccall("extern air.reverse_bits.i32", llvmcall, Int32, (Int32,), x)
@device_function reverse_bits(x::UInt32)  = ccall("extern air.reverse_bits.i32", llvmcall, UInt32, (UInt32,), x)
@device_function reverse_bits(x::Int16)   = ccall("extern air.reverse_bits.i16", llvmcall, Int16, (Int16,), x)
@device_function reverse_bits(x::UInt16)  = ccall("extern air.reverse_bits.i16", llvmcall, UInt16, (UInt16,), x)
@device_function reverse_bits(x::Int8)    = ccall("extern air.reverse_bits.i8", llvmcall, Int8, (Int8,), x)
@device_function reverse_bits(x::UInt8)   = ccall("extern air.reverse_bits.i8", llvmcall, UInt8, (UInt8,), x)


function _mulhi(a::Int64, b::Int64)
    shift = sizeof(a) * 4
    mask = typemax(UInt32)
    a1, a2 = (a >> shift), a & mask
    b1, b2 = (b >> shift), b & mask
    a1b1, a1b2, a2b1 = a1*b1, a1*b2, a2*b1
    t1 = a1b2 + _mulhi(a2 % UInt32, b2 % UInt32)
    t2 = a2b1 + (t1 & mask)
    a1b1 + (t1 >> shift) + (t2 >> shift)
end
@static if isdefined(Base.MultiplicativeInverses, :_mul_high)
    _mulhi(a::T, b::T) where {T<:Union{Signed, Unsigned}} = Base.MultiplicativeInverses._mul_high(a, b)
    @device_override Base.MultiplicativeInverses._mul_high(a::Int64, b::Int64) = _mulhi(a, b)
else
    _mulhi(a::T, b::T) where {T<:Union{Signed, Unsigned}} = ((widen(a)*b) >>> (sizeof(a)*8)) % T
    @device_override function Base.div(a::Int64, b::Base.MultiplicativeInverses.SignedMultiplicativeInverse{Int64})
        x = _mulhi(a, b.multiplier)
        x += (a*b.addmul) % Int64
        ifelse(abs(b.divisor) == 1, a*b.divisor, (signbit(x) + (x >> b.shift)) % Int64)
    end
end

# From: https://forums.developer.nvidia.com/t/a-faster-and-more-accurate-implementation-of-expm1f/48085/2
# Original license copied below:
#  Copyright (c) 2015-2023 Norbert Juffa
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#  1. Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

@device_override function Base.expm1(a::Float32)
    # exp(a) = 2**i * exp(f); i = rintf (a / log(2))
    j = fma(1.442695f0, a, 12582912.0f0)
    j = j - 12582912.0f0
    i = unsafe_trunc(Int32, j)
    f = fma(j, -6.93145752f-1, a) # log_2_hi
    f = fma(j, -1.42860677f-6, f) # log_2_lo

    # approximate r = exp(f)-1 on interval [-log(2)/2, +log(2)/2]
    s = f * f
    if a == 0.0f0
        s = a # ensure -0 is passed through
    end

    # err = 0.997458  ulp1 = 11094007
    r = fma(1.98423862f-4, f, 1.39347673f-3)
    t = fma(8.33342969f-3, f, 4.16667424f-2)
    r = fma(r, s, t)
    r = fma(r, f, 1.66666701f-1)
    r = fma(r, f, 4.99999970f-1)
    # if i == 0, expm1(a) = z
    # if i == 1, expm1(a) = 2*(r*(f*f)+f+0.5)
    # if (i < 0) || (i > 1) expm1(a) = 2*((r*(f*f)+f)*t-0.5+t)
    u = (j == 1) ? (f + 0.5f0) : f
    v = fma(r, s, u)
    s = 0.5f0
    t = ldexp(s, i)
    y = t - s
    x = (t - y) - s # double-float canonicalization of difference
    r = fma(v, t, x) + y
    r = r + r

    if j == 0
        r = v
    end

    if j == 1
        r = v + v
    end

    # handle severe overflow and underflow
    if abs(a - 1.0f0) > 88.0f0
        r = 2^a
        r = fma(r, r, -1.0f0)
    end

    return r
end