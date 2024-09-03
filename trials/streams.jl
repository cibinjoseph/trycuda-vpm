using CUDA

function kernel(x)
    tid = threadIdx().x + blockDim().x * (blockIdx().x-1)
    j = 1
    while j <= 10^4
        x[tid] = sqrt(3.14159^3)
        j += 1
    end
    return
end

function main(n)
    a = ones(n)
    b = ones(n)
    c = ones(n)
    d = ones(n)

    @sync begin
        @async begin
            a_d = CuArray(a)
            @cuda threads=256 blocks=2 kernel(a_d)
            a .= Array(a_d)
        end

        @async begin
            b_d = CuArray(b)
            @cuda threads=256 blocks=2 kernel(b_d)
            b .= Array(b_d)
        end

        # @async begin
        #     c_d = CuArray(c)
        #     @cuda threads=256 blocks=2 kernel(c_d)
        #     c .= Array(c_d)
        # end
        #
        # @async begin
        #     d_d = CuArray(d)
        #     @cuda threads=256 blocks=2 kernel(d_d)
        #     d .= Array(d_d)
        # end
    end
    return
end

n = 2^18

main(n)
CUDA.@profile main(n)
