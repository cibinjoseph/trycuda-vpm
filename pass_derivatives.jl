using FiniteDiff
using ForwardDiff

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

r = 5.0
gamma = 2.0
df_ad = ForwardDiff.gradient(get_net_interaction, [r, gamma])
df_fd = FiniteDiff.finite_difference_gradient(get_net_interaction, [r, gamma])
@assert isapprox(df_ad, df_fd; atol=tol)
