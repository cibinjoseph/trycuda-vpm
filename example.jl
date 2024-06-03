using CUDA
using BenchmarkTools

# Simple vector addition
function cpu_add1!(y, x)
    @inbounds y .+= x
    return nothing
end

function gpu_add1!(y, x)
    idx = threadIdx().x + (blockIdx().x-1)*blockDim().x
    stride = blockDim().x * gridDim().x
    for i = idx:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

# Simple matrix addition
function cpu_addmat!(y, x)
    @inbounds y .+= x
    return nothing
end

function gpu_addmat!(y, x)
    idx_x = threadIdx().x + (blockIdx().x-1)*blockDim().x
    stride_x = blockDim().x * gridDim().x
    idx_y = threadIdx().y + (blockIdx().y-1)*blockDim().y
    stride_y = blockDim().y * gridDim().y

    for j = idx_y:stride_y:size(y, 2)
        for i = idx_x:stride_x:size(y, 1)
            @inbounds y[i, j] += x[i, j]
        end
    end

    return nothing
end

# Simple single vector from matrix
function cpu_vel1!(v, y, x)
    for j = 1:size(x, 2)
        @inbounds v[1, j] = y[1, 1] - x[1, j]
        @inbounds v[2, j] = y[2, 1] - x[2, j]
        @inbounds v[3, j] = y[3, 1] - x[3, j]
        @inbounds v[4, j] = v[1, j]^2 + v[2, j]^2 + v[3, j]^2
        @inbounds v[5, j] = v[1, j] / v[4, j]
    end
    return nothing
end

function gpu_vel1!(v, y, x)
    idx = threadIdx().x + (blockIdx().x-1)*blockDim().x
    stride = blockDim().x * gridDim().x

    for j = idx:stride:size(x, 2)
        @inbounds v[1, j] = y[1, 1] - x[1, j]
        @inbounds v[2, j] = y[2, 1] - x[2, j]
        @inbounds v[3, j] = y[3, 1] - x[3, j]
        @inbounds v[4, j] = v[1, j]^2 + v[2, j]^2 + v[3, j]^2
        @inbounds v[5, j] = v[1, j] / v[4, j]
    end

    return nothing
end

# benchmark functions
function bench_add1!(y, x)
    nblocks = ceil(Int, length(y)/256)
    CUDA.@sync begin
        @cuda threads=256 blocks=nblocks gpu_add1!(y, x)
    end
end

function bench_addmat!(y, x, nblocks_x, nblocks_y)
    CUDA.@sync begin
        @cuda threads=(32, 32) blocks=(nblocks_x, nblocks_y) gpu_addmat!(y, x)
    end
end

function bench_addmat_mem!(y, x, nblocks_x, nblocks_y)
    x_d = CuArray(x)
    y_d = CuArray(y)
    CUDA.@sync begin
        @cuda threads=(32, 32) blocks=(nblocks_x, nblocks_y) gpu_addmat!(y_d, x_d)
    end
end

function bench_vel1!(v, y, x, nblocks_x, nblocks_y)
    CUDA.@sync begin
        @cuda threads=32 blocks=nblocks_y gpu_vel1!(v, y, x)
    end
end

function bench_vel1_mem!(v, y, x, nblocks_x, nblocks_y)
    x_d = CuArray(x)
    y_d = CuArray(y)
    v_d = CuArray(v)
    CUDA.@sync begin
        @cuda threads=32 blocks=nblocks_y gpu_vel1!(v_d, y_d, x_d)
    end
end

function cpu_mul!(c, a, b)
    for i in 1:size(a, 1)
        for j in 1:size(b, 2)
            for k in 1:size(a, 2)
                c[i, j] += a[i, k] * b[k, j]
            end
        end
    end
    return nothing
end

function gpu_mul1!(c, a, b)
    idx::Int32 = threadIdx().x + (blockIdx().x-1)*blockDim().x
    idy::Int32 = threadIdx().y + (blockIdx().y-1)*blockDim().y
    k::Int32 = 1
    while k <= size(a, 2)
        c[idx, idy] += a[idx, k] * b[k, idy]
        k += 1
    end

    return nothing
end

function gpu_mul2!(c, a, b)
    idx::Int32 = threadIdx().x + (blockIdx().x-1)*blockDim().x
    idy::Int32 = threadIdx().y + (blockIdx().y-1)*blockDim().y
    k::Int32 = 1
    while k <= size(a, 2)
        c[idx, idy] += a[idx, k] * b[k, idy]
        k += 1
    end

    return nothing
end

function bench_gpu_mul1!(c, a, b)
    c_d = CuArray(c)
    a_d = CuArray(a)
    b_d = CuArray(b)
    m = size(a, 1)
    n = size(b, 2)
    bx = ceil(Int32, m/32)
    by = ceil(Int32, n/32)
    CUDA.@sync begin
        @cuda threads=(32, 32) blocks=(bx, by) gpu_mul1!(c_d, a_d, b_d)
    end
    c = Array(c_d)
    return nothing
end

# check array ops inside kernel
function cpu_array_kernel!(v, y, x)
    v .= x.*v
    # y .+= reduce(+, x, dims=2)
    for i in 1:size(y, 2)
        for j in 1:size(x, 2)
            y[i] += x[j]
        end
    end
    return nothing
end

function gpu_array_kernel!(v, y, x)
    v .= map(*, x, v)
    y .+= reduce(+, x, dims=2)
    return nothing
end

# n = 2^12
# nf = 43
# T = Float32

# x = rand(T, nf, n)
# y = rand(T, nf, n)
# @btime cpu_addmat!(y, x)
#
# x_d = CuArray(x)
# y_d = CuArray(y)
# nblocks_x = ceil(Int, nf/32)
# nblocks_y = ceil(Int, n/32)
# @btime bench_addmat!(y_d, x_d, nblocks_x, nblocks_y)

# x = rand(T, nf, n)
# y = rand(T, nf, 1)
# v = rand(T, nf, n)
# @btime cpu_vel1!(v, y, x)

# x_d = CuArray(x)
# y_d = CuArray(y)
# v_d = CuArray(v)
# nblocks_x = 1
# nblocks_y = ceil(Int, n/32)
# @btime bench_vel1!(v_d, y_d, x_d, nblocks_x, nblocks_y)
# @btime bench_vel1_mem!(v, y, x, nblocks_x, nblocks_y)

# x_d = CuArray(x)
# y_d = CuArray(y)
# v_d = CuArray(v)
# @btime cpu_array_kernel!($v, $y, $x)

# kernel = @cuda launch=false gpu_array_kernel!(v_d, y_d, x_d)
# CUDA.launch_configuration(kernel.fun)

# @btime CUDA.@sync gpu_array_kernel!($v_d, $y_d, $x_d)
# CUDA.@profile trace=true gpu_array_kernel!(v_d, y_d, x_d)

m = 2^12
n = 2^12
k = 2^12
T = Float32

a = rand(T, m, k)
b = rand(T, k, n)
c1 = zeros(T, m, n)
c2 = zeros(T, m, n)

@btime cpu_mul!(c1, a, b)
@btime bench_gpu_mul1!(c2, a, b)
