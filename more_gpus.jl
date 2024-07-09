using CUDA
using Random

@inline function f!(mat)
    j = threadIdx().x
    i = 1
    while i <= 4
        mat[i, j] = i + 1
        i += 1
    end
    return
end


n = 2^11
T = Float64
mat_cpu = rand(T, 4, n)

n1 = cld(n, 2)
n2 = n - n1

ndevices = length(devices())
if ndevices == 2
    @sync begin
        @async begin
            # Device 1
            device!(0)
            mat_gpu1 = CuArray(view(mat_cpu, 1:4, 1:n1))
            threads = n1
            blocks = 1
            @cuda threads=threads blocks=blocks f!(mat_gpu1)
            mat_cpu[:, 1:n1] .= Array(mat_gpu1)
        end

        @async begin
            # Device 2
            device!(1)
            mat_gpu2 = CuArray(view(mat_cpu, 1:4, n1+1:n))
            threads = n2
            blocks = 1
            @cuda threads=threads blocks=blocks f!(mat_gpu2)
            mat_cpu[:, n1+1:n] .= Array(mat_gpu2)
        end
    end
else
    println("Only $ndevices found")
end
