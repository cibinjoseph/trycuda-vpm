using CUDA

function kernel(x, i)
    tid = threadIdx().x + blockDim().x * (blockIdx().x-1)
    j = 1
    while j <= 10^4
        x[tid, i] = sqrt(3.14159^3)
        j += 1
    end
    return
end

function main(a, nstreams)
    a_d = CuArray{eltype(a)}(undef, length(a), nstreams)

    for i=1:nstreams
        stream = CuStream()
        @cuda threads=256 blocks=1 stream=stream kernel(a_d, i)

        # Single stream
        # @cuda threads=256 blocks=1 kernel(a_d, i)
    end
    return
end

n = 2^20
nstreams = 8
a = ones(n)

main(a, nstreams)
CUDA.@profile main(a, nstreams)
