using FiniteDiff
using ForwardDiff

include("vpm.jl")

tol = 1f-4  # Difference b/w finite diff and forward diff

function get_net_interaction_cpu(x::Vector{T}) where T
    radius = x[1]
    gammaX = x[2]
    gammaY = x[3]
    gammaZ = x[4]
    sigma  = x[5]

    nparticles, nfields = 10, 43

    theta = LinRange{Float32}(0, 2*pi, nparticles+1)[1:end-1]

    src = zeros(T, nfields, nparticles)

    # Construct source bodies in circle
    for i in 1:nparticles
        src[1, i] = radius * cos(theta[i])
        src[2, i] = radius * sin(theta[i])
        src[4, i] = gammaX
        src[5, i] = gammaY
        src[6, i] = gammaZ
        src[7, i] = sigma
    end

    cpu_vpm!(src, src)

    vel = zero(T)
    for j in 1:nparticles
        for i in 5:7
            vel += src[i, j]^2
        end
    end

    return vel
end

function cpu_func!(y, x)
    @show x
    for i = 1:length(x)
        y[i] = x[i]*x[i] + 2
    end
    return
end

function gpu_func!(y, x)
    idx::Int32 = threadIdx().x + (blockIdx().x-1)*blockDim().x

    @cushow typeof(x)

    if idx <= length(x)
        y[idx] = x[idx]*x[idx] + 2.0f0
    end
    return
end

function gpu_runner!(y::Vector{T}, x::Vector{T}) where T
    x_d = CuArray(view(x, :, :))
    y_d = similar(x_d)
    kernel = gpu_g_dgdr
    n = length(x)
    p, q = get_launch_config(n; T=T)
    nthreads::Int32 = p*q
    nblocks::Int32 = cld(n, p)
    CUDA.@sync @cuda threads=nthreads blocks=nblocks gpu_vpm3!(y_d, x_d, q, kernel)
    y[10:12] .= Array(y_d[10:12, :])
    y[16:24] .= Array(y_d[16:24, :])
    return
end

# CPU gravitation kernel
T = Float64
x = rand(T, 5)
x_gpu = deepcopy(x)
@show get_net_interaction_cpu(x)
@show df_ad = ForwardDiff.gradient(get_net_interaction_cpu, x)
# @show df_fd = FiniteDiff.finite_difference_gradient(get_net_interaction_cpu, x)
# @assert isapprox(df_ad, df_fd; atol=tol)

# GPU kernel - simple
# n = 2^2
# x = rand(Float32, n)
# y = similar(x)
# df_cpu = ForwardDiff.jacobian(cpu_func!, y, x)
# df_gpu = ForwardDiff.jacobian(gpu_runner!, y, x)
# @assert isapprox(df_cpu, df_gpu; atol=tol)

# GPU gravitation kernel
function get_net_interaction_gpu(x::Vector{T}) where T
    radius = x[1]
    gammaX = x[2]
    gammaY = x[3]
    gammaZ = x[4]
    sigma  = x[5]

    nparticles, nfields = 10, 43

    theta = LinRange(0, 2*pi, nparticles+1)[1:end-1]

    src = zeros(T, nfields, nparticles)

    # Construct source bodies in circle
    for i in 1:nparticles
        src[1, i] = radius * cos(theta[i])
        src[2, i] = radius * sin(theta[i])
        src[4, i] = gammaX
        src[5, i] = gammaY
        src[6, i] = gammaZ
        src[7, i] = sigma
    end

    s_d = CuArray(view(src, 1:24, :))


    kernel = gpu_g_dgdr
    p, q = get_launch_config(nparticles; T=T)
    nthreads::Int32 = p*q
    nblocks::Int32 = cld(nparticles, p)
    shmem = sizeof(T) * (12*p) * p

    # Check if shared memory is sufficient
    dev = CUDA.device()
    dev_shmem = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    if shmem > dev_shmem
        error("Shared memory requested exceeds available space on GPU")
    end
    CUDA.@sync @cuda threads=nthreads blocks=nblocks shmem=shmem gpu_vpm5!(s_d, s_d, q, kernel)
    view(src, 10:12, :) .= Array(s_d[10:12, :])
    view(src, 16:24, :) .= Array(s_d[16:24, :])

    vel = zero(T)
    for j in 1:nparticles
        for i in 5:7
            vel += src[i, j]^2
        end
    end

    return vel
end

@show get_net_interaction_gpu(x_gpu)

cfg1 = ForwardDiff.GradientConfig(get_net_interaction_gpu, x_gpu, ForwardDiff.Chunk{4}());
@show df_ad = ForwardDiff.gradient(get_net_interaction_gpu, x_gpu, cfg1)
# @show df_fd = FiniteDiff.finite_difference_gradient(get_net_interaction_gpu, x_gpu)
# @assert isapprox(df_ad, df_fd; atol=tol)
