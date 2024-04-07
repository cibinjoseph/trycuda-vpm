using Random
using CUDA
using BenchmarkTools

include("pfield.jl")

# Winckelmans algebraic kernel
function g_dgdr(r)
    aux0 = (r^2 + 1)^2.5
    # Returns g, dgdr
    return r^3 * (r^2 + 2.5) / aux0, 7.5 * r^2 / (aux0*(r^2 + 1))
end

# Winckelmans algebraic kernel
g_val(r) = r^3 * (r^2 + 2.5) / (r^2 + 1)^2.5
dg_val(r) = 7.5 * r^2 / ((r^2 + 1)^2.5*(r^2 + 1))

function UJ_direct(sources, targets, g_dgdr::Function)
    r = zero(eltype(sources.particles))
    for Pi in iterate(targets)
        for Pj in iterate(sources)

            dX1 = get_X(Pi)[1] - get_X(Pj)[1]
            dX2 = get_X(Pi)[2] - get_X(Pj)[2]
            dX3 = get_X(Pi)[3] - get_X(Pj)[3]
            r2 = dX1*dX1 + dX2*dX2 + dX3*dX3

            if !iszero(r2)
                r = sqrt(r2)

                # Regularizing function and deriv
                g_sgm, dg_sgmdr = g_dgdr(r/get_sigma(Pj)[])

                # K × Γp
                crss1 = -const4 / r^3 * ( dX2*get_Gamma(Pj)[3] - dX3*get_Gamma(Pj)[2] )
                crss2 = -const4 / r^3 * ( dX3*get_Gamma(Pj)[1] - dX1*get_Gamma(Pj)[3] )
                crss3 = -const4 / r^3 * ( dX1*get_Gamma(Pj)[2] - dX2*get_Gamma(Pj)[1] )

                # U = ∑g_σ(x-xp) * K(x-xp) × Γp
                get_U(Pi)[1] += g_sgm * crss1
                get_U(Pi)[2] += g_sgm * crss2
                get_U(Pi)[3] += g_sgm * crss3

                # ∂u∂xj(x) = ∑[ ∂gσ∂xj(x−xp) * K(x−xp)×Γp + gσ(x−xp) * ∂K∂xj(x−xp)×Γp ]
                # ∂u∂xj(x) = ∑p[(Δxj∂gσ∂r/(σr) − 3Δxjgσ/r^2) K(Δx)×Γp
                aux = dg_sgmdr/(get_sigma(Pj)[]*r) - 3*g_sgm /r^2
                # j=1
                get_J(Pi)[1] += aux * crss1 * dX1
                get_J(Pi)[2] += aux * crss2 * dX1
                get_J(Pi)[3] += aux * crss3 * dX1
                # j=2
                get_J(Pi)[4] += aux * crss1 * dX2
                get_J(Pi)[5] += aux * crss2 * dX2
                get_J(Pi)[6] += aux * crss3 * dX2
                # j=3
                get_J(Pi)[7] += aux * crss1 * dX3
                get_J(Pi)[8] += aux * crss2 * dX3
                get_J(Pi)[9] += aux * crss3 * dX3

                # ∂u∂xj(x) = −∑gσ/(4πr^3) δij×Γp
                # Adds the Kronecker delta term
                aux = - const4 * g_sgm / r^3

                # j=1
                get_J(Pi)[2] -= aux * get_Gamma(Pj)[3]
                get_J(Pi)[3] += aux * get_Gamma(Pj)[2]
                # j=2
                get_J(Pi)[4] += aux * get_Gamma(Pj)[3]
                get_J(Pi)[6] -= aux * get_Gamma(Pj)[1]
                # j=3
                get_J(Pi)[7] -= aux * get_Gamma(Pj)[2]
                get_J(Pi)[8] += aux * get_Gamma(Pj)[1]

            end
        end
    end
    return nothing
end

# Map reduce version
function UJ_direct_map(sources, targets, g_dgdr::Function)
    for target in iterate(targets)
        UJ_direct_map1(sources, target, g_dgdr)
    end
    return nothing
end

cross3(a, b) = [a[2]*b[3]-a[3]*b[2], a[3]*b[1]-a[1]*b[3], a[1]*b[2]-a[2]*b[1]]

function UJ_direct_map1(sources, Pi, g_dgdr)
    dX = view(Pi, 1:3) .- view(sources.particles, 1:3, :)
    r2 = mapreduce(x->x^2, +, dX, dims=1)
    r = map(sqrt, r2)
    r3 = map(x->x^3, r)

    # Regularizing function and deriv
    rbysigma = map(/, r, view(sources.particles, 7, :))
    g_sgm = map(g_val, rbysigma)
    dg_sgmdr = map(dg_val, rbysigma)

    # K × Γp = -Γp × K
    crss = zeros(size(dX))
    for i in 1:size(dX, 2)
        crss[:, i] = cross3(dX[:, i], view(sources.particles, 4:6, i))
    end
    @views crss .= -const4*crss./r3

    @views aux = dg_sgmdr./map(*, r, view(sources.particles, 7, :)) .- 3*map(/, g_sgm, r2)
    # dX[1, :] .*= aux
    # dX[2, :] .*= aux
    # dX[3, :] .*= aux
    @views dX .= aux' .* dX
    @views aux1 = - const4 * map(/, g_sgm, r3)

    # U = ∑g_σ(x-xp) * K(x-xp) × Γp
    @views Pi[10:12] .+= reduce(+, g_sgm' .* crss, dims=2)

    # ∂u∂xj(x) = ∑[ ∂gσ∂xj(x−xp) * K(x−xp)×Γp + gσ(x−xp) * ∂K∂xj(x−xp)×Γp ]
    # ∂u∂xj(x) = ∑p[(Δxj∂gσ∂r/(σr) − 3Δxjgσ/r^2) K(Δx)×Γp
    # j=1
    @views Pi[16:18] .+= reduce(+, dX[1, :]' .* crss, dims=2)
    # j=2
    @views Pi[19:21] .+= reduce(+, dX[2, :]' .* crss, dims=2)
    # j=3
    @views Pi[22:24] .+= reduce(+, dX[3, :]' .* crss, dims=2)

    # ∂u∂xj(x) = −∑gσ/(4πr^3) δij×Γp
    @views Jterm1 = reduce(+, aux1 .* sources.particles[4, :])
    @views Jterm2 = reduce(+, aux1 .* sources.particles[5, :])
    @views Jterm3 = reduce(+, aux1 .* sources.particles[6, :])

    # j=1
    @views Pi[17] -= Jterm3
    @views Pi[18] += Jterm2
    # j=2
    @views Pi[19] += Jterm3
    @views Pi[21] -= Jterm1
    # j=3
    @views Pi[22] -= Jterm2
    @views Pi[23] += Jterm1
    return nothing
end

function UJ_simple(sources, targets, g_dgdr::Function)
    for Pi in iterate(targets)
        for Pj in iterate(sources)
            dX1 = get_X(Pi)[1] - get_X(Pj)[1]
            dX2 = get_X(Pi)[2] - get_X(Pj)[2]
            dX3 = get_X(Pi)[3] - get_X(Pj)[3]
            r2 = dX1*dX1 + dX2*dX2 + dX3*dX3
            get_X(Pj)[1] = dX1 / r2 
            get_X(Pj)[2] = dX2 / r2 
            get_X(Pj)[3] = dX3 / r2 
        end
    end
end

function UJ_simple_gpu(sources, targets, g_dgdr::Function)
    src = GPUParticleField(sources)
    trg = GPUParticleField(targets)
    for Pi in iterate(src)
        for Pj in iterate(trg)
            dX = get_X(Pi) - get_X(Pj)
            r2 = sum(dX.^2)
            get_X(Pj) .= dX ./ r2 
        end
    end
    targets.particles[1:3, :] .= Array(trg.particles[1:3, :])
    return nothing
end

function UJ_simple_gpu2(sources, targets, g_dgdr::Function)
    dX = CUDA.zeros(3)
    for i in 1:size(sources.particles, 2)
        for j in 1:size(targets.particles, 2)
            dX .= get_X(sources, i) - get_X(targets, j)
            r2 = sum(dX.^2)
            get_X(targets, i) .= dX ./ r2 
        end
    end
    return nothing
end

nparticles = 5
Random.seed!(1234)
mat1_orig = rand(nfields, nparticles)
mat2_orig = rand(nfields, nparticles)

sources = ParticleField(nparticles, mat1_orig)
targets = ParticleField(nparticles, mat2_orig)

sources2 = ParticleField(nparticles, deepcopy(mat1_orig))
targets2 = ParticleField(nparticles, deepcopy(mat2_orig))

UJ_direct(sources, targets, g_dgdr)
UJ_direct_map(sources2, targets2, g_dgdr)

# @btime UJ_simple(sources, targets, g_dgdr)

# GPU part 1
# sources2 = ParticleField(nparticles, mat1_orig)
# targets2 = ParticleField(nparticles, mat2_orig)
# @btime UJ_simple_gpu(sources2, targets2, g_dgdr)

# GPU part 2
# sources2 = GPUParticleField(sources)
# targets2 = GPUParticleField(targets)
# @btime UJ_simple_gpu2(sources2, targets2, g_dgdr)
# CUDA.@profile trace=true UJ_simple_gpu2(sources2, targets2, g_dgdr)

# Verify they are the same
if !all(abs.(targets.particles .- Array(targets2.particles)) .< 1E-10)
    println("MISMATCH !!!")
end

# @btime UJ_direct(sources, targets, g_dgdr)
# @btime UJ_direct_map(sources2, targets2, g_dgdr)
