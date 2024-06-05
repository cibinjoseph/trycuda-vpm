using CUDA
using BenchmarkTools
using SpecialFunctions

# CPU
const const2 = sqrt(2/pi)
const sqr2 = sqrt(2)
function cpu_g_dgdr(r::T) where T
    aux::T = T(const2)*r*exp(-r^2/2)
    g::T = SpecialFunctions.erf(r/T(sqr2))-aux
    dg::T = r*aux
    return g, dg
end

# GPU
# erf constants
const erf_a1 =  0.254829592
const erf_a2 = -0.284496736
const erf_a3 =  1.421413741
const erf_a4 = -1.453152027
const erf_a5 =  1.061405429
const erf_p  =  0.3275911
@inline function my_erf(x::Float64)
    # Abramowitz & Stegen, formula 7.1.26
    # Max error is below 1e-7
    xabs = abs(x)
    t = 1.0/(1.0 + erf_p*xabs)
    y = 1.0 - (((((erf_a5*t + erf_a4)*t) + erf_a3)*t + erf_a2)*t + erf_a1)*t*exp(-xabs*xabs)
    return sign(x)*y
end

const erff_a1 =  0.254829592f0
const erff_a2 = -0.284496736f0
const erff_a3 =  1.421413741f0
const erff_a4 = -1.453152027f0
const erff_a5 =  1.061405429f0
const erff_p  =  0.3275911f0
@inline function my_erf(x::Float32)
    # Abramowitz & Stegen, formula 7.1.26
    # Max error is below 1e-7
    xabs = abs(x)
    t = 1.0f0/(1.0f0 + erff_p*xabs)
    y = 1.0f0 - (((((erff_a5*t + erff_a4)*t) + erff_a3)*t + erff_a2)*t + erff_a1)*t*exp(-xabs*xabs)
    return sign(x)*y
end

# CUDA erf function
@inline Cuerf(x::Float64) = ccall("extern __nv_erf", llvmcall, Cdouble, (Cdouble,), x)
@inline Cuerf(x::Float32) = ccall("extern __nv_erff", llvmcall, Cfloat, (Cfloat,), x)

function gpu_g_dgdr(r)
    aux = const2*r*exp(-r^2/2)
    return my_erf(r/sqr2)-aux, r*aux
end

# n = 2^3
# T = Float32
#
# r = rand(T, n)
# g = zeros(T, n)
# dg = zeros(T, n)
#
# r_d = CuArray(r)
# g_d = CuArray(g)
# dg_d = CuArray(dg)
#
# cpu_g_dgdr!(g, dg, r)
# @cuda threads=length(r) gpu_g_dgdr!(g_d, dg_d, r_d)
