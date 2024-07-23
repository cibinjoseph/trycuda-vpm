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

# Function to demonstrate parallelization when there are multiple leaves and gpus
function direct(nleaves, ngpus)
    ileaf = 1
    while ileaf <= nleaves
        nleaves_remaining = nleaves-ileaf+1 
        # Perform computations on GPU
        ileaf_gpu = ileaf
        for ig in min(ngpus, nleaves_remaining):-1:1
            println("GPU $(ig-1) computes leaf $ileaf_gpu")
            ileaf_gpu += 1
        end

        # Copy results back to CPU
        ileaf_gpu = ileaf
        for ig in min(ngpus, nleaves_remaining):-1:1
            println("GPU $(ig-1) obtains leaf $ileaf_gpu")
            ileaf_gpu += 1
        end
        ileaf = ileaf_gpu
    end
end

n = 2^13
T = Float64
mat_cpu = rand(T, 400, n)

function work!(mat_cpu)
    ndevices = length(devices())
    println("$ndevices GPU found")

    if ndevices == 1
        # mat_gpu = CuArray(mat_cpu)
        # threads = min(n, 1024)
        # blocks = cld(n, threads)
        # @cuda threads=threads blocks=blocks f!(mat_gpu)
        # mat_cpu .= Array(mat_gpu)

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

function kernel!(a)
    j = threadIdx().x
    a[j] = 1.0
    return
end

function run_kernel!(a)
    idx = [1:10, 25:30]

    ndevices = length(CUDA.devices())

    a_gpu_list = Vector{CuArray{Float64, 1}}(undef, ndevices)

    # Launch kernels on multiple gpus
    for idev in 1:ndevices
        CUDA.device!(idev-1)
        a_gpu = CuArray(view(a, idx[idev]))
        n = length(idx[idev])
        @cuda threads=n kernel!(a_gpu)
        a_gpu_list[idev] = a_gpu
        @show size(a_gpu)
    end

    # Copy results back from gpus
    for idev in 1:ndevices
        CUDA.device!(idev-1)
        a[idx[idev]] .= Array(a_gpu_list[idev])
        @show size(a_gpu_list[idev])
    end

end

# work!(mat_cpu)
# CUDA.@profile work!(mat_cpu)

a = zeros(30)
run_kernel!(a)
