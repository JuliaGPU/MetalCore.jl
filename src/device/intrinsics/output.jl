const MTLLOG_SUBSYSTEM = "com.juliagpu.metal.jl"
const MTLLOG_CATEGORY = "mtlprintf"

const __METAL_OS_LOG_TYPE_DEBUG__ = Int32(2)
const __METAL_OS_LOG_TYPE_INFO__ = Int32(1)
const __METAL_OS_LOG_TYPE_DEFAULT__ = Int32(0)
const __METAL_OS_LOG_TYPE_ERROR__ = Int32(16)
const __METAL_OS_LOG_TYPE_FAULT__ = Int32(17)

export @mtlprintf

@generated function promote_c_argument(arg)
    # > When a function with a variable-length argument list is called, the variable
    # > arguments are passed using C's old ``default argument promotions.'' These say that
    # > types char and short int are automatically promoted to int, and type float is
    # > automatically promoted to double. Therefore, varargs functions will never receive
    # > arguments of type char, short int, or float.

    if arg == Cchar || arg == Cshort
        return :(Cint(arg))
    elseif arg == Cfloat
        return :(Cdouble(arg))
    else
        return :(arg)
    end
end

function valist_size(dl, param_types)
    size = 0
    for pty in param_types
        ps = sizeof(dl, pty)
        if size % ps == 0
            size += ps
        else
            size += (size % ps) + ps
        end
    end

    return size
end

"""
    @mtlprintf("%Fmt", args...)

Print a formatted string in device context on the host standard output.
"""
macro mtlprintf(fmt::String, args...)
    fmt_val = Val(Symbol(fmt))

    quote
        if metal_version() < sv"3.2"
            @print_and_throw "@mtlprintf requires Metal 3.2 (macOS 15) or higher"
        else
            _mtlprintf($fmt_val, $(map(arg -> :(promote_c_argument($arg)), esc.(args))...))
        end
    end
end

@generated function _mtlprintf(::Val{fmt}, argspec...) where {fmt}
    @dispose ctx=Context() begin
        arg_exprs = [:( argspec[$i] ) for i in 1:length(argspec)]
        arg_types = [argspec...]

        T_void = LLVM.VoidType()
        T_int32 = LLVM.Int32Type()
        T_int64 = LLVM.Int64Type()
        T_pint8 = LLVM.PointerType(LLVM.Int8Type())
        T_pint8a2 = LLVM.PointerType(LLVM.Int8Type(), 2)

        # create functions
        param_types = LLVMType[convert(LLVMType, typ) for typ in arg_types]
        wrapper_f, wrapper_ft = create_function(T_void, param_types)
        mod = LLVM.parent(wrapper_f)

        llvm_ft = LLVM.FunctionType(T_void, LLVMType[]; vararg=true)
        llvm_f = LLVM.Function(mod, "metal_os_log", llvm_ft)
        push!(function_attributes(llvm_f), EnumAttribute("alwaysinline", 0))

        # generate IR
        @dispose builder=IRBuilder() begin
            entry = BasicBlock(llvm_f, "entry")
            position!(builder, entry)

            str = globalstring_ptr!(builder, String(fmt), addrspace=2)
            subsystem_str = globalstring_ptr!(builder, MTLLOG_SUBSYSTEM, addrspace=2)
            category_str = globalstring_ptr!(builder, MTLLOG_CATEGORY, addrspace=2)
            log_type = LLVM.ConstantInt(T_int32, __METAL_OS_LOG_TYPE_DEBUG__)

            # compute argsize
            dl = datalayout(mod)
            arg_size = LLVM.ConstantInt(T_int64, valist_size(dl, param_types))

            alloc = alloca!(builder, T_pint8)
            buffer = bitcast!(builder, alloc, T_pint8)
            alloc_size = LLVM.ConstantInt(T_int64, sizeof(dl, T_pint8))

            lifetime_start_fty = LLVM.FunctionType(T_void, [T_int64, T_pint8])
            lifetime_start = LLVM.Function(mod, "llvm.lifetime.start.p0i8", lifetime_start_fty)
            call!(builder, lifetime_start_fty, lifetime_start, [alloc_size, buffer])

            va_start_fty = LLVM.FunctionType(T_void, [T_pint8])
            va_start = LLVM.Function(mod, "llvm.va_start", va_start_fty)
            call!(builder, va_start_fty, va_start, [buffer])

            arg_ptr = load!(builder, T_pint8, alloc)

            os_log_fty = LLVM.FunctionType(T_void, [T_pint8a2, T_pint8a2, T_int32, T_pint8a2, T_pint8, T_int64])
            os_log = LLVM.Function(mod, "air.os_log", os_log_fty)
            call!(builder, os_log_fty, os_log, [subsystem_str, category_str, log_type, str, arg_ptr, arg_size])

            va_end_fty = LLVM.FunctionType(T_void, [T_pint8])
            va_end = LLVM.Function(mod, "llvm.va_end", va_end_fty)
            call!(builder, va_end_fty, va_end, [buffer])

            lifetime_end_fty = LLVM.FunctionType(T_void, [T_int64, T_pint8])
            lifetime_end = LLVM.Function(mod, "llvm.lifetime.end.p0i8", lifetime_end_fty)
            call!(builder, lifetime_end_fty, lifetime_end, [alloc_size, buffer])

            ret!(builder)
        end

        @dispose builder=IRBuilder() begin
            entry = BasicBlock(wrapper_f, "entry")
            position!(builder, entry)

            call!(builder, llvm_ft, llvm_f, collect(parameters(wrapper_f)))

            ret!(builder)
        end


        call_function(wrapper_f, Nothing, Tuple{arg_types...}, arg_exprs...)
    end
end


## print-like functionality

export @mtlprint, @mtlprintln

# simple conversions, defining an expression and the resulting argument type. nothing fancy,
# `@mtlprint` pretty directly maps to `@mtlprintf`; we should just support `write(::IO)`.
const mtlprint_conversions = [
    Float32         => (x->:(Float64($x)),          Float64),
    Ptr{<:Any}      => (x->:(reinterpret(Int, $x)), Ptr{Cvoid}),
    LLVMPtr{<:Any}  => (x->:(reinterpret(Int, $x)), Ptr{Cvoid}),
    Bool            => (x->:(Int32($x)),            Int32),
]

# format specifiers
const mtlprint_specifiers = Dict(
    # integers
    Int16       => "%hd",
    Int32       => "%d",
    Int64       => "%ld",
    UInt16      => "%hu",
    UInt32      => "%u",
    UInt64      => "%lu",

    # floating-point
    Float32     => "%f",

    # other
    Cchar       => "%c",
    Ptr{Cvoid}  => "%p",
    Cstring     => "%s",
)

@inline @generated function _mtlprint(parts...)
    fmt = ""
    args = Expr[]

    for i in 1:length(parts)
        part = :(parts[$i])
        T = parts[i]

        # put literals directly in the format string
        if T <: Val
            fmt *= string(T.parameters[1])
            continue
        end

        # try to convert arguments if they are not supported directly
        if !haskey(mtlprint_specifiers, T)
            for (Tmatch, rule) in mtlprint_conversions
                if T <: Tmatch
                    part = rule[1](part)
                    T = rule[2]
                    break
                end
            end
        end

        # render the argument
        if haskey(mtlprint_specifiers, T)
            fmt *= mtlprint_specifiers[T]
            push!(args, part)
        elseif T <: Tuple
            fmt *= "("
            for (j, U) in enumerate(T.parameters)
                if haskey(mtlprint_specifiers, U)
                    fmt *= mtlprint_specifiers[U]
                    push!(args, :($part[$j]))
                    if j < length(T.parameters)
                        fmt *= ", "
                    elseif length(T.parameters) == 1
                        fmt *= ","
                    end
                else
                    @error("@mtlprint does not support values of type $U")
                end
            end
            fmt *= ")"
        elseif T <: String
            @error("@mtlprint does not support non-literal strings")
        elseif T <: Type
            fmt *= string(T.parameters[1])
        else
            @warn("@mtlprint does not support values of type $T")
            fmt *= "$(T)(...)"
        end
    end

    quote
        @mtlprintf($fmt, $(args...))
    end
end

"""
    @mtlprint(xs...)
    @mtlprintln(xs...)

Print a textual representation of values `xs` to standard output from the GPU. The
functionality builds on `@mtlprintf`, and is intended as a more use friendly alternative of
that API. However, that also means there's only limited support for argument types, handling
16/32/64 signed and unsigned integers, 32 and 64-bit floating point numbers, `Cchar`s and
pointers. For more complex output, use `@mtlprintf` directly.

Limited string interpolation is also possible:

```julia
    @mtlprint("Hello, World ", 42, "\\n")
    @mtlprint "Hello, World \$(42)\\n"
```
"""
macro mtlprint(parts...)
    args = Union{Val,Expr,Symbol}[]

    parts = [parts...]
    while true
        isempty(parts) && break

        part = popfirst!(parts)

        # handle string interpolation
        if isa(part, Expr) && part.head == :string
            parts = vcat(part.args, parts)
            continue
        end

        # expose literals to the generator by using Val types
        if isbits(part) # literal numbers, etc
            push!(args, Val(part))
        elseif isa(part, QuoteNode) # literal symbols
            push!(args, Val(part.value))
        elseif isa(part, String) # literal strings need to be interned
            push!(args, Val(Symbol(part)))
        else # actual values that will be passed to printf
            push!(args, part)
        end
    end

    quote
        _mtlprint($(map(esc, args)...))
    end
end

@doc (@doc @mtlprint) ->
macro mtlprintln(parts...)
    esc(quote
        Metal.@mtlprint($(parts...), "\n")
    end)
end

export @mtlshow

"""
    @mtlshow(ex)

GPU analog of `Base.@show`. It comes with the same type restrictions as [`@mtlprintf`](@ref).

```julia
@mtlshow thread_position_in_grid_1d()
```
"""
macro mtlshow(exs...)
    blk = Expr(:block)
    for ex in exs
        push!(blk.args, :(Metal.@mtlprintln($(sprint(Base.show_unquoted,ex)*" = "),
                                          begin local value = $(esc(ex)) end)))
    end
    isempty(exs) || push!(blk.args, :value)
    blk
end
