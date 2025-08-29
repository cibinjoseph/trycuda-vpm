using FiniteDiff
using ForwardDiff
using Random
using Plots
using Polynomials
using BenchmarkTools
using DelimitedFiles

Random.seed!(1)

include("vpm.jl")
include("benchmark_extras.jl")

tol = 1f-4  # Difference b/w finite diff and forward diff
const nfields = 43
const kernel = g_dgdr_wnklmns

function get_coeffs(ncoeffs=16)
    n = 100
    b = 8
    x_ellipse = LinRange(0, b, n)
    y_ellipse = @. sqrt(1-(x_ellipse*x_ellipse)/b^2)

    deg = ncoeffs-1
    p = fit(x_ellipse, y_ellipse, deg)

    return x_ellipse, y_ellipse, p.coeffs
end

# Assigns coordinate positions as a carpet extending from an elliptic curve
function create_distribution!(src, coeffs; nx=16)
    n = size(src, 2)
    b = 8
    x = LinRange(0, b, nx)
    p = Polynomial(coeffs)
    y = p.(x)
    Δy = 0.1

    ny = cld(n, nx)
    nn = 1
    for i in 1:ny
        for j in 1:nx
            @inbounds src[1, nn] = x[j]
            @inbounds src[2, nn] = y[j] - Δy*(i-1)
            nn += 1
        end
    end
    return x, y
end

# CPU interaction kernel
function get_net_interaction_cpu(x::Vector{T}, nparticles::Int) where T
    coeffs = x
    gammaX = 0.5
    gammaY = 0.5
    gammaZ = 0.5
    sigma  = 0.05

    src = zeros(T, nfields, nparticles)

    # Construct source bodies in circle
    create_distribution!(src, coeffs)
    @inbounds begin
        for i in 1:nparticles
            src[4, i] = gammaX
            src[5, i] = gammaY
            src[6, i] = gammaZ
            src[7, i] = sigma
        end
    end

    cpu_vpm!(src, src)

    vel = zero(T)
    for j in 1:nparticles
        @inbounds vel += sqrt(src[10, j]^2 + src[11, j]^2 + src[12, j]^2)
    end

    return vel
end

# VPM gravitation kernel
function get_net_interaction_gpu(x::Vector{T}, nparticles::Int) where T
    coeffs = x
    gammaX = 0.5
    gammaY = 0.5
    gammaZ = 0.5
    sigma  = 0.05

    src = zeros(T, nfields, nparticles)

    # Construct source bodies in circle
    create_distribution!(src, coeffs)
    @inbounds begin
        for i in 1:nparticles
            src[4, i] = gammaX
            src[5, i] = gammaY
            src[6, i] = gammaZ
            src[7, i] = sigma
        end
    end

    s_d = CuArray(view(src, 1:7, :))
    t_d = CuArray(view(src, 1:3, :))
    out = CUDA.zeros(T, 12, nparticles)

    # Verify max threads per block limit
    p, q = optimal_pq(nparticles)
    k = @cuda launch=false gpu_vpm11!(out, s_d, t_d, p, q, kernel)
    maxthreads = CUDA.maxthreads(k)
    if maxthreads < 512
        @warn "Max threads lowered to $maxthreads due to high register pressure"
    end


    p, q = optimal_pq(nparticles; productmax=maxthreads)
    nthreads::Int32 = p*q
    nblocks::Int32 = cld(nparticles, p)
    shmem = sizeof(T) * 8 * p

    # Check if shared memory is sufficient
    dev = CUDA.device()
    dev_shmem = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    if shmem > dev_shmem
        error("Shared memory requested ($shmem) exceeds available space on GPU ($dev_shmem)")
    end

    @cuda threads=nthreads blocks=nblocks shmem=shmem gpu_vpm11!(out, s_d, t_d, p, q, kernel)
    @inbounds src[10:12, 1:nparticles] .+= Array(out[1:3, 1:nparticles])
    @inbounds src[16:24, 1:nparticles] .+= Array(out[4:12, 1:nparticles])

    vel = zero(T)
    for j in 1:nparticles
        @inbounds vel += sqrt(src[10, j]^2 + src[11, j]^2 + src[12, j]^2)
    end

    return vel
end

# Set up variables
function benchmark_AD(ncoeffs, nparticles)
    xe, ye, coeffs = get_coeffs(ncoeffs)

    x = coeffs
    x_gpu = deepcopy(x)

    # Closures for AD to pass
    cpu_func = x -> get_net_interaction_cpu(x, nparticles)
    gpu_func = x -> get_net_interaction_gpu(x, nparticles)

    ############
    # CPU call #
    ############
    # @show vel = cpu_func(x)
    vel = cpu_func(x)
    result_cpu = @benchmark $cpu_func($x)
    t_cpu = median(result_cpu.times) / 1e9

    df_ad = ForwardDiff.gradient(cpu_func, x)
    result_cpuAD = @benchmark ForwardDiff.gradient($cpu_func, $x)
    t_cpuAD = median(result_cpuAD.times) / 1e9

    # @show df_fd = FiniteDiff.finite_difference_gradient(cpu_func, x)
    # @assert isapprox(df_ad, df_fd; atol=tol)

    ############
    # GPU call #
    ############
    vel = gpu_func(x_gpu)
    result_gpu = @benchmark $gpu_func($x_gpu)
    t_gpu = median(result_gpu.times) / 1e9

    # chunk_size = ForwardDiff.pickchunksize(length(x))
    # @info "Auto chunk size = $chunk_size"
    # cfg1 = ForwardDiff.GradientConfig(gpu_func, x_gpu, ForwardDiff.Chunk{1}());
    df_ad = ForwardDiff.gradient(gpu_func, x_gpu)
    result_gpuAD = @benchmark ForwardDiff.gradient($gpu_func, $x_gpu)
    t_gpuAD = median(result_gpuAD.times) / 1e9

    # @show df_fd = FiniteDiff.finite_difference_gradient(gpu_func, x_gpu)
    # @assert isapprox(df_ad, df_fd; atol=tol)
    #

    return  ncoeffs, nparticles, t_cpu, t_gpu, t_cpuAD, t_gpuAD
end


# Quick check
# nparticles = 2^4 * 2^12
# ncoeffs = 20

# # Closures
# cpu_func = x -> get_net_interaction_cpu(x, nparticles)
# gpu_func = x -> get_net_interaction_gpu(x, nparticles)

# xe, ye, coeffs = get_coeffs(ncoeffs)

# x = coeffs
# x_gpu = deepcopy(x)
# vel = gpu_func(x_gpu)

# df_ad = ForwardDiff.gradient(gpu_func, x_gpu)

# Benchmarking
ncoeffs_list = 1:20
nparticles_list = 2^4 * (2 .^ collect(6:16))
filename = "output.csv"

data = zeros(6)

header = "ncoeffs nparticles t_cpu t_gpu t_cpuAD t_gpuAD"
ensure_header(filename, header)

for nparticles in nparticles_list, ncoeffs in ncoeffs_list
    if case_exists(filename, ncoeffs, nparticles)
        @info "Skipping case: $ncoeffs, $nparticles"
    else
        data .= benchmark_AD(ncoeffs, nparticles)
        append_row(filename, data)
    end
end

