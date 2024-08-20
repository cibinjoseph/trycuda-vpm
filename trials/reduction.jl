using CUDA
using BenchmarkTools
using Primes
using StaticArrays


n = 2^12
T = Float64
a_cpu = ones(T, n)
s_cpu = zeros(T, 1)
a_gpu = deepcopy(a_cpu)
s_gpu = deepcopy(s_cpu)

function cpu_sum!(s, a)
    for i in 1:length(a)
        s[1] += a[i]
    end
    return
end

# Atomic reduction
function sum_reduction1!(sb, a)
    ithread::Int32 = threadIdx().x + blockDim().x*(blockIdx().x-1)

    sh_mem = CuDynamicSharedArray(eltype(a), blockDim().x)

    # Each thread copies element to shared memory
    sh_mem[threadIdx().x] = a[ithread]

    # Each thread sums to sh_mem[1]
    if threadIdx().x != 1
        CUDA.@atomic sh_mem[1] += sh_mem[threadIdx().x]
    end
    sync_threads()

    if threadIdx().x == 1
        sb[blockIdx().x] = sh_mem[1]
    end
    return
end

# Atomic reduction runner
function gpu_sum1!(s::Vector{T}, a, blocks) where T
    a_d = CuArray(a)
    sb = CUDA.zeros(blocks)

    threads = cld(length(a), blocks)
    shmem_1 = sizeof(T) * threads
    shmem_2 = sizeof(T) * blocks
    @cuda threads=threads blocks=blocks shmem=shmem_1 sum_reduction1!(sb, a_d)

    @cuda threads=blocks blocks=1 shmem=shmem_2 sum_reduction1!(sb, sb)
    s .= Array(view(sb, 1))
end

# Shared memory parallel reduction, sequential threads
function sum_reduction2!(sb, a)
    ithread::Int32 = threadIdx().x + blockDim().x*(blockIdx().x-1)

    sh_mem = CuDynamicSharedArray(eltype(a), blockDim().x)

    # Each thread copies element to shared memory
    sh_mem[threadIdx().x] = a[ithread]
    sync_threads()

    # Sum up every element inside a block to sb
    stride::Int32 = 1
    while stride < blockDim().x
        it::Int32 = (threadIdx().x-1)*stride*2+1
        if it <= blockDim().x
            sh_mem[it] += sh_mem[it+stride]
        end
        stride *= 2
        sync_threads()
    end

    # Copy first element to sb
    if threadIdx().x == 1
        sb[blockIdx().x] = sh_mem[1]
    end

    return
end

# Shared memory parallel reduction, sequential threads runner
function gpu_sum2!(s::Vector{T}, a, blocks) where T
    a_d = CuArray(a)
    sb = CUDA.zeros(blocks)

    threads = cld(length(a), blocks)
    shmem_1 = sizeof(T) * threads
    shmem_2 = sizeof(T) * blocks
    @cuda threads=threads blocks=blocks shmem=shmem_1 sum_reduction2!(sb, a_d)

    @cuda threads=blocks blocks=1 shmem=shmem_2 sum_reduction2!(sb, sb)
    s .= Array(view(sb, 1))
end

# Shared memory parallel reduction, sequential threads, contiguous access
function sum_reduction3!(sb, a)
    ithread::Int32 = threadIdx().x + blockDim().x*(blockIdx().x-1)

    sh_mem = CuDynamicSharedArray(eltype(a), blockDim().x)

    # Each thread copies element to shared memory
    sh_mem[threadIdx().x] = a[ithread]
    sync_threads()

    # Sum up every element inside a block to sb
    stride::Int32 = blockDim().x/2
    while stride > 0
        it::Int32 = threadIdx().x
        if it+stride <= blockDim().x
            sh_mem[it] += sh_mem[it+stride]
        end
        stride = CUDA.floor(Int32, stride/2)
        sync_threads()
    end

    # Copy first element to sb
    if threadIdx().x == 1
        sb[blockIdx().x] = sh_mem[1]
    end

    return
end

function gpu_sum3!(s::Vector{T}, a, blocks) where T
    a_d = CuArray(a)
    sb = CUDA.zeros(blocks)

    threads = cld(length(a), blocks)
    shmem_1 = sizeof(T) * threads
    shmem_2 = sizeof(T) * blocks
    @cuda threads=threads blocks=blocks shmem=shmem_1 sum_reduction3!(sb, a_d)

    @cuda threads=blocks blocks=1 shmem=shmem_2 sum_reduction3!(sb, sb)
    s .= Array(view(sb, 1))
end

# Shared memory parallel reduction, sequential threads, contiguous access, half threads
function sum_reduction4!(sb, a)
    ithread::Int32 = threadIdx().x + blockDim().x*(blockIdx().x-1)

    sh_mem = CuDynamicSharedArray(eltype(a), blockDim().x)

    # Each thread copies element and first reduction-level to shared memory
    if blockDim().x > 1
        sh_mem[threadIdx().x] = a[ithread] + a[ithread+blockDim().x]
    else
        sh_mem[threadIdx().x] = a[ithread]
    end
    sync_threads()

    # Sum up every element inside a block to sb
    stride::Int32 = blockDim().x/2
    while stride > 0
        it::Int32 = threadIdx().x
        if it+stride <= blockDim().x
            sh_mem[it] += sh_mem[it+stride]
        end
        stride = CUDA.floor(Int32, stride/2)
        sync_threads()
    end

    # Copy first element to sb
    if threadIdx().x == 1
        sb[blockIdx().x] = sh_mem[1]
    end

    return
end

function gpu_sum4!(s::Vector{T}, a, blocks) where T
    a_d = CuArray(a)
    sb = CUDA.zeros(blocks)

    threads = cld(length(a), 2*blocks)
    shmem_1 = sizeof(T) * threads
    @cuda threads=threads blocks=blocks shmem=shmem_1 sum_reduction4!(sb, a_d)

    threads = cld(blocks, 2)
    shmem_2 = sizeof(T) * threads
    @cuda threads=threads blocks=1 shmem=shmem_2 sum_reduction4!(sb, sb)
    s .= Array(view(sb, 1))
end

function benchmark_gpu!(s, a, choice=1)
    div = divisors(length(a))
    blocks = div[cld(length(div), 2)]
    if choice == 1
        CUDA.@sync gpu_sum1!(s, a, blocks)
    elseif choice == 2
        CUDA.@sync gpu_sum2!(s, a, blocks)
    elseif choice == 3
        CUDA.@sync gpu_sum3!(s, a, blocks)
    elseif choice == 4
        CUDA.@sync gpu_sum4!(s, a, blocks)
    end
    return
end

@btime cpu_sum!(s_cpu, a_cpu)
@btime benchmark_gpu!(s_gpu, a_gpu, 1)
@btime benchmark_gpu!(s_gpu, a_gpu, 2)
@btime benchmark_gpu!(s_gpu, a_gpu, 3)
@btime benchmark_gpu!(s_gpu, a_gpu, 4)

# CUDA.@sync gpu_sum4!(s_gpu, a_gpu, 4)

# CUDA.@profile benchmark1_gpu!(s_gpu, a_gpu)
