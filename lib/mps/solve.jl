
export MPSMatrixSolveTriangular

@objcwrapper immutable=false MPSMatrixSolveTriangular <: MPSMatrixUnaryKernel

function MPSMatrixSolveTriangular(device, right, upper, unit, order, numberOfRightHandSides, alpha)
    kernel = @objc [MPSMatrixSolveTriangular alloc]::id{MPSMatrixSolveTriangular}
    obj = MPSMatrixSolveTriangular(kernel)
    finalizer(release, obj)
    @objc [obj::id{MPSMatrixSolveTriangular} initWithDevice:device::id{MTLDevice}
                                             right:right::Bool
                                             upper:upper::Bool
                                             transpose:transpose::Bool
                                             unit:unit::Bool
                                             order:order::NSUInteger
                                             numberOfRightHandSides:numberOfRightHandSides::NSUInteger
                                             alpha:alpha::Float64]::id{MPSMatrixSolveTriangular}
    return obj
end

function encode!(cmdbuf::MTLCommandBuffer, kernel::MPSMatrixSolveTriangular, sourceMatrix, resultMatrix, pivotIndices, status)
    @objc [kernel::id{MPSMatrixSolveTriangular} encodeToCommandBuffer:cmdbuf::id{MTLCommandBuffer}
                                                sourceMatrix:sourceMatrix::id{MPSMatrix}
                                                resultMatrix:resultMatrix::id{MPSMatrix}
                                                pivotIndices:pivotIndices::id{MPSMatrix}
                                                status:status::id{MPSMatrix}]::Nothing
end


export MPSMatrixSolveLU

@objcwrapper immutable=false MPSMatrixSolveLU <: MPSMatrixUnaryKernel

function MPSMatrixSolveLU(device, transpose, order, numberOfRightHandSides)
    kernel = @objc [MPSMatrixSolveLU alloc]::id{MPSMatrixSolveLU}
    obj = MPSMatrixSolveLU(kernel)
    finalizer(release, obj)
    @objc [obj::id{MPSMatrixSolveLU} initWithDevice:device::id{MTLDevice}
                                             transpose:transpose::Bool
                                             order:order::NSUInteger
                                             numberOfRightHandSides:numberOfRightHandSides::NSUInteger]::id{MPSMatrixSolveLU}
    return obj
end

function encode!(cmdbuf::MTLCommandBuffer, kernel::MPSMatrixSolveLU, sourceMatrix, rightHandSideMatrix, pivotIndices, solutionMatrix)
    @objc [kernel::id{MPSMatrixSolveLU} encodeToCommandBuffer:cmdbuf::id{MTLCommandBuffer}
                                                sourceMatrix:sourceMatrix::id{MPSMatrix}
                                                rightHandSideMatrix:rightHandSideMatrix::id{MPSMatrix}
                                                pivotIndices:pivotIndices::id{MPSMatrix}
                                                solutionMatrix:solutionMatrix::id{MPSMatrix}]::Nothing
end




export MPSMatrixSolveCholesky

@objcwrapper immutable=false MPSMatrixSolveCholesky <: MPSMatrixUnaryKernel

function MPSMatrixSolveCholesky(device, upper, order, numberOfRightHandSides)
    kernel = @objc [MPSMatrixSolveCholesky alloc]::id{MPSMatrixSolveCholesky}
    obj = MPSMatrixSolveCholesky(kernel)
    finalizer(release, obj)
    @objc [obj::id{MPSMatrixSolveCholesky} initWithDevice:device::id{MTLDevice}
                                             upper:upper::Bool
                                             order:order::NSUInteger
                                             numberOfRightHandSides:numberOfRightHandSides::NSUInteger]::id{MPSMatrixSolveCholesky}
    return obj
end

function encode!(cmdbuf::MTLCommandBuffer, kernel::MPSMatrixSolveCholesky, sourceMatrix, rightHandSideMatrix, solutionMatrix)
    @objc [kernel::id{MPSMatrixSolveCholesky} encodeToCommandBuffer:cmdbuf::id{MTLCommandBuffer}
                                                sourceMatrix:sourceMatrix::id{MPSMatrix}
                                                rightHandSideMatrix:rightHandSideMatrix::id{MPSMatrix}
                                                pivotIndices:pivotIndices::id{MPSMatrix}
                                                solutionMatrix:solutionMatrix::id{MPSMatrix}]::Nothing
end
