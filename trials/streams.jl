using CUDA

function kernel(x, n)
    tid = threadIdx().x + blockDim().x * (blockIdx().x-1)
    i = tid
    while i <= n
        x[i] = sqrt(3.14159^i)
        i += blockDim().x * gridDim().x
    end
    return
end

n = 2^20
nstreams = 8
a = ones(n)
a_d = CuArray{eltype(a)}(undef, n, nstreams)

for i=1:nstreams
    stream = CuStream()
    @cuda threads=64 blocks=1 stream=stream kernel(a_d, n)
end
