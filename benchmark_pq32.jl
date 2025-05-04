using Primes
using StaticArrays
include("vpm.jl")
include("divs_512.jl")

# Read input arg if available
n_default = 5000
n = length(ARGS) == 0 ? n_default : parse(Int, ARGS[1])

# Assemble divisor pairs that result in 32 as product multiple
pqs = []
for i = 1:size(divs_512, 2)
    p = divs_512[1, i]
    q = divs_512[2, i]
    push!(pqs, (p, q))
end

for (p, q) in pqs
    _, _, _, _, speedup = main(3; nt=n, ns=n, p=p, q=q, debug=false, algorithm=7, padding=false, show_pq=true, return_vals=true)
end
