using CUDA
using Random
using NVTX

@inline function kernel!(a)
    i::Int32 = threadIdx().x + blockDim().x*(blockIdx().x-1)
    @inbounds a[i] = CUDA.cos(a[i]) + i^0.6 + CUDA.tan(a[i])
    return
end

function main()
    n = 2^13
    a = rand(n)
    
    threads = 256
    blocks = cld(n, threads)
    CUDA.@sync a_d = CuArray(a)
    NVTX.@mark "Start of kernel call"
    CUDA.@sync @cuda threads=threads blocks=blocks kernel!(a_d)
    NVTX.@mark "End of kernel call"
    a = Array(a_d)

    return
end

main()
CUDA.@profile main()
