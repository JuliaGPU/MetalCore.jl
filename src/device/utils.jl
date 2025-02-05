# local method table for device functions
Base.Experimental.@MethodTable(method_table)

macro print_and_throw(args...)
    quote
        #@println "ERROR: " $(args...) "."
        throw(nothing)
    end
end

macro device_override(ex)
    ex = macroexpand(__module__, ex)
    esc(quote
        Base.Experimental.@overlay($method_table, $ex)
    end)
end

macro device_function(ex)
    ex = macroexpand(__module__, ex)
    def = splitdef(ex)

    # generate a function that errors
    def[:body] = quote
        error("This function is not intended for use on the CPU")
    end

    esc(quote
        $(combinedef(def))
        @device_override $ex
    end)
end
