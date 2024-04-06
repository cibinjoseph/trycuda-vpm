using CUDA
using BenchmarkTools

include("pfield.jl")

# Winckelmans algebraic kernel
function g_dgdr(r)
    aux0 = (r^2 + 1)^2.5
    # Returns g, dgdr
    return r^3 * (r^2 + 2.5) / aux0, 7.5 * r^2 / (aux0*(r^2 + 1))
end

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

nparticles = 100
mat1_orig = rand(nfields, nparticles)
mat2_orig = rand(nfields, nparticles)

sources = ParticleField(nparticles, mat1_orig)
targets = ParticleField(nparticles, mat2_orig)

@btime UJ_simple(sources, targets, g_dgdr)

# GPU part 1
# sources2 = ParticleField(nparticles, mat1_orig)
# targets2 = ParticleField(nparticles, mat2_orig)
# @btime UJ_simple_gpu(sources2, targets2, g_dgdr)

# GPU part 2
sources2 = GPUParticleField(sources)
targets2 = GPUParticleField(targets)
@btime UJ_simple_gpu2(sources2, targets2, g_dgdr)
# CUDA.@profile trace=true UJ_simple_gpu2(sources2, targets2, g_dgdr)

# Verify they are the same
# if !all(targets.particles .== Array(targets2.particles))
#     println("MISMATCH !!!")
# end
