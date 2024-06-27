using CUDA
using BenchmarkTools


n = 2^5
T = Float64
a = ones(T, n)
s = zeros(T, 1)

function cpu_sum!(s, a)
    for i in 1:length(a)
        s[1] += a[i]
        @show s
    end
    return
end

function sum_reduction!(sb, s)
    ithread = threadIDx().x + (blockDim().x-1)*blockIdx.x

    # Each thread copy one element to shared memory
    sh_mem[threadIDx().x] = s[ithread]

    # Sum up every element inside a block to sb
    for i in 1:2:blockDim().x
        sb[blockIdx().x]
    end

    return
end

function gpu_sum!(s, a, blocks)
    s_d = CuArray(s)
    sb = CUDA.zeros(nblocks)

    threads = cld(length(s), blocks)
    @cuda threads=threads blocks=blocks sum_reduction!(sb, s)
    @cuda threads=nblocks 
end
