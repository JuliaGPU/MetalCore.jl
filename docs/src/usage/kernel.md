# Kernel programming

Metal.jl is based off of Apple's [Metal Shading Language (MSL)](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
and Metal framework. The interface allows you to utilize the graphics and computing power of
Mac GPUs. Like many other GPU frameworks, its history is rooted in graphics processing but
has found use in computing/general purpose GPU (GPGPU) applications.

The most fundamental idea of programming GPUs (when compared to serial CPU programming) is
its *parallelism*. A GPU function (kernel), when called, is not just ran once in isolation.
Rather, numerous (often thousands to millions) psuedo-independent instances (called threads)
of the kernel are executed in parallel. These threads are arranged in a hierarchy that allows
for varying levels of synchronization. For Metal, the hierarchy is as follows:

- Thread: A single execution unit of the kernel
- Threadgroup: A collection of threads that share a common block of memory and synchronization
barriers
- Grid: A collection of threadgroups

The threadgroup and grid sizes are set by the user when launching the GPU kernel. There are
upper limits determined by the targeted hardware, and the sizes can be 1, 2, or 3-dimensional. For
Metal.jl, these sizes are set using the `@metal` macro's keyword arguments. The `grid`
keyword determines the grid size while the `threads` keyword determines the threadgroup size.

For example, given a 10x10x3 image that you want to run a function independently on each pixel,
the kernel launch code might look like the following:
`@metal threads=(10,10) groups=3 my_kernel(gpu_image_array)`
This would launch 3 separate threadgroups of 100 threads each (10 in the first dimension and
10 in the second dimension)

There is also additional hierarchy layers that consists of small groups of threads
that execute in lockstep called *waves/SIMD groups/*wavefronts* and *quadgroups*. However, the basic
three-tier hierarchy is enough to get started.

[Here](https://developer.apple.com/documentation/metal/compute_passes/creating_threads_and_threadgroups?language=objc)
is a helpful link with good visualizations of Metal's thread hierarchy (also covering
SIMD groups).

Each thread has its own set of private variables. Most importantly, each thread has
associated unique indices to identify itself within its threadgroup and grid.
These are traditionally what are used to differentiate execution across threads. You can
also query what the grid and threadgroup sizes are as well.

For Metal.jl, these values are accessed via the following functions:

- `thread_index_in_threadgroup()`
- `grid_size_Xd()`
- `thread_position_in_grid_Xd()`
- `thread_position_in_threadgroup_Xd()`
- `threadgroup_position_in_grid_Xd()`
- `threadgroups_per_grid_Xd()`
- `threads_per_grid_Xd()`
- `threads_per_threadgroup_Xd()`

*Where 'X' is 1, 2, or 3 according to the number of dimensions requested.*

Using these in a kernel (taken directly from the [vadd example](https://github.com/JuliaGPU/Metal.jl/blob/main/examples/vadd.jl)):

```julia
function vadd(a, b, c)
    i = thread_position_in_grid_1d()
    c[i] = a[i] + b[i]
    return
end
```

This kernel takes in three vectors (a,b,c) all of the same length and stores the element-wise
sum of `a` and `b` into `c`. Each thread in this kernel gets its unique position in the grid
(arrangement of all threadgroups) and stores this value into the variable `i` which is then
used as the index into the vectors. Thus, each thread is computing one sum and storing the result
in the output vector.

To ensure this kernel functions properly, we have to launch it with exactly as many threads
as the length of the vectors. If we under or over-launch threads, the result could be incorrect.

An example of a good launch:

```julia
len = prod(size(d_a))
@metal threads=len vadd(d_a, d_b, d_c)
```

Additional notes:

- Kernels must always return nothing
- Kernels are asynchronous. To synchronize, use the `Metal.@sync` macro.

## Printing

When debugging, it's not uncommon to want to print some values. This is achieved with `@mtlprintf`:

```julia
function gpu_add2_print!(y, x)
    index = thread_position_in_grid_1d()
    @mtlprintf("thread %d", index)
    @inbounds y[i] += x[i]
    return nothing
end

A = Metal.ones(Float32, 8);
B = Metal.rand(Float32, 8);

@metal threads=length(A) gpu_add2_print!(A, B)
```

`@mtlprintf` is supported on macOS 15 and later. `@mtlprintf` support most of the format specifiers that `printf`
supports in C with the following exceptions:
 - `%n` and `%s` conversion specifiers are not supported
 - Default argument promotion applies to arguments of half type which promote to the `double` type
 - The format string must be a string literal

Metal places output from `@mtlprintf` into a log buffer. The system only removes the messages from the log buffer when the command buffer completes. When the log buffer becomes full, the system drops all subsequent messages.

See also: `@mtlprint`, `@mtlprintln` and `@mtlshow`

## Other Helpful Links

[Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
