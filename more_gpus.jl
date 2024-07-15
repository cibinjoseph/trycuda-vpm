using CUDA
using Random

@inline function f!(mat)
    j = threadIdx().x + blockDim().x * (blockIdx().x-1)
    i = 1
    while i <= size(mat, 1)
        k = 1
        while k <= 100
            mat[i, j] = i + 1 + sin(i/k)
            k += 1
        end
        i += 1
    end
    return
end


n = 2^13
T = Float64
mat_cpu = rand(T, 400, n)

function work!(mat_cpu)
    ndevices = length(devices())
    println("$ndevices GPU found")

    if ndevices == 1
        mat_gpu = CuArray(mat_cpu)
        threads = min(n, 1024)
        blocks = cld(n, threads)
        @cuda threads=threads blocks=blocks f!(mat_gpu)
        mat_cpu .= Array(mat_gpu)

    elseif ndevices >= 2
        # Launch kernels on gpu/s
        istart = 1
        n_remain = n
        for i in ndevices:-1:1
            # Device 1
            device!(i-1)
            istop = istart + floor(Int, n_remain/i) - 1
            n_ele = istop-istart+1
            # @show istart, istop
            mat_gpu = CuArray(view(mat_cpu, 1:size(mat_cpu, 1), istart:istop))
            threads = min(n_ele, 1024)
            blocks = cld(n_ele, threads)
            @cuda threads=threads blocks=blocks f!(mat_gpu)

            # Update index
            n_remain -= n_ele
            istart = istop + 1
        end

        # Copy back results from gpu/s
        istart = 1
        n_remain = n
        for i in ndevices:-1:1
            device!(i-1)
            istop = istart + floor(Int, n_remain/i) - 1
            mat_cpu[:, istart:istop] .= Array(mat_gpu)

            # Update index
            n_remain -= istop - istart + 1
            istart = istop + 1
        end
    else
        println("$ndevices GPU devices found")
    end
end

work!(mat_cpu)
CUDA.@profile work!(mat_cpu)
