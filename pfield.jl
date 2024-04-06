using CUDA

nfields = 43
const const4 = 1/(4*pi)

mutable struct ParticleField{R<:Real}
    maxparticles::Int
    particles::Matrix{R}
end

mutable struct GPUParticleField{R<:Real}
    maxparticles::Int
    particles::CuArray{R}
end

function ParticleField(maxparticles::Int, R=Float64)
    particles = rand(R, nfields, maxparticles)
    return ParticleField(maxparticles, particles)
end

function ParticleField(maxparticles::Int, particles)
    return ParticleField(maxparticles, particles)
end

function GPUParticleField(pfield::ParticleField)
    return GPUParticleField(pfield.maxparticles, CuArray(pfield.particles))
end

Base.iterate(pfield::ParticleField) = _get_particleiterator(pfield)
Base.iterate(pfield::GPUParticleField) = _get_particleiterator(pfield)

_get_particleiterator(pfield::ParticleField) = eachcol(view(pfield.particles, :, :))
_get_particleiterator(pfield::GPUParticleField) = eachcol(view(pfield.particles, :, :))


"Get functions for particles"
# This is (and should be) the only place that explicitly
# maps the indices of each particle's fields
get_X(P) = view(P, 1:3)
get_Gamma(P) = view(P, 4:6)
get_sigma(P) = view(P, 7)
get_vol(P) = view(P, 8)
get_circulation(P) = view(P, 9)
get_U(P) = view(P, 10:12)
get_vorticity(P) = view(P, 13:15)
get_J(P) = view(P, 16:24)
get_PSE(P) = view(P, 25:27)
get_M(P) = view(P, 28:36)
get_C(P) = view(P, 37:39)
get_SFS(P) = view(P, 40:42)
get_static(P) = view(P, 43)

is_static(P) = Bool(P[43])

# This extra function computes the vorticity using the cross-product
get_W(P) = (get_W1(P), get_W2(P), get_W3(P))

get_W1(P) = get_J(P)[6]-get_J(P)[8]
get_W2(P) = get_J(P)[7]-get_J(P)[3]
get_W3(P) = get_J(P)[2]-get_J(P)[4]

get_SFS1(P) = get_SFS(P)[1]
get_SFS2(P) = get_SFS(P)[2]
get_SFS3(P) = get_SFS(P)[3]

get_particle(pfield, i::Int) = view(pfield.particles, :, i)

"Get functions for particles in ParticleField"
get_X(pfield::ParticleField, i::Int) = get_X(get_particle(pfield, i))
get_X(pfield::GPUParticleField, i::Int) = get_X(get_particle(pfield, i))
get_Gamma(pfield::ParticleField, i::Int) = get_Gamma(get_particle(pfield, i))
get_sigma(pfield::ParticleField, i::Int) = get_sigma(get_particle(pfield, i))
get_vol(pfield::ParticleField, i::Int) = get_vol(get_particle(pfield, i))
get_circulation(pfield::ParticleField, i::Int) = get_circulation(get_particle(pfield, i))
get_U(pfield::ParticleField, i::Int) = get_U(get_particle(pfield, i))
get_vorticity(pfield::ParticleField, i::Int) = get_vorticity(get_particle(pfield, i))
get_J(pfield::ParticleField, i::Int) = get_J(get_particle(pfield, i))
get_PSE(pfield::ParticleField, i::Int) = get_PSE(get_particle(pfield, i))
get_W(pfield::ParticleField, i::Int) = get_W(get_particle(pfield, i))
get_M(pfield::ParticleField, i::Int) = get_M(get_particle(pfield, i))
get_C(pfield::ParticleField, i::Int) = get_C(get_particle(pfield, i))
get_static(pfield::ParticleField, i::Int) = get_static(get_particle(pfield, i))

is_static(pfield::ParticleField, i::Int) = is_static(get_particle(pfield, i))

"Set functions for particles"
set_X(P, val) = get_X(P) .= val
set_Gamma(P, val) = get_Gamma(P) .= val
set_sigma(P, val) = get_sigma(P) .= val
set_vol(P, val) = get_vol(P) .= val
set_circulation(P, val) = get_circulation(P) .= val
set_U(P, val) = get_U(P) .= val
set_vorticity(P, val) = get_vorticity(P) .= val
set_J(P, val) = get_J(P) .= val
set_M(P, val) = get_M(P) .= val
set_C(P, val) = get_C(P) .= val
set_static(P, val) = get_static(P) .= val
set_PSE(P, val) = get_PSE(P) .= val
set_SFS(P, val) = get_SFS(P) .= val

"Set functions for particles in ParticleField"
set_X(pfield::ParticleField, i::Int, val) = set_X(get_particle(pfield, i), val)
set_Gamma(pfield::ParticleField, i::Int, val) = set_Gamma(get_particle(pfield, i), val)
set_sigma(pfield::ParticleField, i::Int, val) = set_sigma(get_particle(pfield, i), val)
set_vol(pfield::ParticleField, i::Int, val) = set_vol(get_particle(pfield, i), val)
set_circulation(pfield::ParticleField, i::Int, val) = set_circulation(get_particle(pfield, i), val)
set_U(pfield::ParticleField, i::Int, val) = set_U(get_particle(pfield, i), val)
set_vorticity(pfield::ParticleField, i::Int, val) = set_vorticity(get_particle(pfield, i), val)
set_J(pfield::ParticleField, i::Int, val) = set_J(get_particle(pfield, i), val)
set_M(pfield::ParticleField, i::Int, val) = set_M(get_particle(pfield, i), val)
set_C(pfield::ParticleField, i::Int, val) = set_C(get_particle(pfield, i), val)
set_static(pfield::ParticleField, i::Int, val) = set_static(get_particle(pfield, i), val)
set_PSE(pfield::ParticleField, i::Int, val) = set_PSE(get_particle(pfield, i), val)
set_SFS(pfield::ParticleField, i::Int, val) = set_SFS(get_particle(pfield, i), val)
