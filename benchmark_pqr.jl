include("vpm.jl")
using Primes

# Read input arg if available
nt_default = 32
ns_default = 5000
nt = length(ARGS) == 0 ? nt_default : parse(Int, ARGS[1])
ns = length(ARGS) == 0 ? ns_default : parse(Int, ARGS[2])

divs_nt = sort(divisors(nt))
divs_ns = sort(divisors(ns))

outmat = zeros(1, 4)

for p in divs_nt
    for r in divs_ns
        for q in divs_ns
            if q <= r && p*q <= 512 && r <= 876
                main(3; ns=ns, nt=nt, p=p, q=q, r=r, show_pq=true, debug=false, algorithm=9, padding=false, return_vals=false)
                # else
                #     println("$n $p $q nan")
            end
        end
    end
end
