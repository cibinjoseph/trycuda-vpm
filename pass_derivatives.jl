using FiniteDiff
using ForwardDiff
using CUDA
using Random

include("gravitation.jl")

tol = 1E-6

function get_net_interaction(x::Vector{T}) where T
    radius = x[1]
    gamma = x[2]

    nparticles, nfields = 10, 7
    src, trg, src2, trg2 = get_inputs(nparticles, nfields)

    theta = LinRange(0, 2*pi, nparticles+1)[1:end-1]

    src = zeros(T, nfields, nparticles)

    # Construct source bodies in circle
    for i in 1:nparticles
        src[1, i] = radius * cos(theta[i])
        src[2, i] = radius * sin(theta[i])
        src[4, i] = gamma * rand(T)
    end

    cpu_gravity!(src, src)

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
    n = length(x)
    nthreads = min(n, 2^5)
    nblocks = ceil(Int32, n/nthreads)
    CUDA.@sync @cuda threads=nthreads blocks=nblocks gpu_func!(y_d, x_d)
    y .= Array(y_d)
    return
end

r = 5.0
gamma = 2.0
# df_ad = ForwardDiff.gradient(get_net_interaction, [r, gamma])
# df_fd = FiniteDiff.finite_difference_gradient(get_net_interaction, [r, gamma])
# @assert isapprox(df_ad, df_fd; atol=tol)

n = 2^2
Random.seed!(1234)
x = rand(Float32, n)
y = similar(x)
df_cpu = ForwardDiff.jacobian(cpu_func!, y, x)
df_gpu = ForwardDiff.jacobian(gpu_runner!, y, x)
@assert isapprox(df_cpu, df_gpu; atol=tol)
