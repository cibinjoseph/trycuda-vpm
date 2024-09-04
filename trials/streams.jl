using CUDA
import NVTX

function kernel(x)
    tid = threadIdx().x + blockDim().x * (blockIdx().x-1)
    j = 1
    while j <= 10^4
        x[tid] = sqrt(3.14159^3) + CUDA.cos(x[tid]) - CUDA.cos(x[tid])
        j += 1
    end
    return
end

NVTX.@annotate "main_func" function main(n)
    a = ones(n)
    b = ones(n)
    c = ones(n)
    d = ones(n)

    @sync begin
        Threads.@spawn begin
            a_d = CuArray(a)
            @cuda threads=256 blocks=3 kernel(a_d)
            a .= Array(a_d)
        end

        Threads.@spawn begin
            b_d = CuArray(b)
            @cuda threads=256 blocks=3 kernel(b_d)
            b .= Array(b_d)
        end

        Threads.@spawn begin
            c_d = CuArray(c)
            @cuda threads=256 blocks=3 kernel(c_d)
            c .= Array(c_d)
        end

        Threads.@spawn begin
            d_d = CuArray(d)
            @cuda threads=256 blocks=3 kernel(d_d)
            d .= Array(d_d)
        end
    end
    return
end

function benchmark()
    n = 2^20
    main(n)
    main(n)
    main(n)
end

benchmark()
CUDA.@profile benchmark()
