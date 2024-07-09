using CUDA
using Random

@inline function f!(mat)
    j = threadIdx().x
    i = 1
    while i <=4
        mat[i, j] = i + 1
        i += 1
    end
    return
end


n = 8
mat_cpu = rand(4, n)

mat_gpu = CuArray(mat_cpu)
threads = n
blocks = 1

@cuda threads=threads blocks=blocks f!(mat_gpu)
mat_cpu .= Array(view(mat_gpu, :, 1:n))
