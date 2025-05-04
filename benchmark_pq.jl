include("vpm.jl")
using Primes

# Read input arg if available
n_default = 5000
n = length(ARGS) == 0 ? n_default : parse(Int, ARGS[1])

divs = sort(divisors(n))


for p in divs
    divs_p = divisors(p)
    for q in divs
        if q in divs_p && p*q <= 512
            main(3; ns=n, p=p, q=q, show_pq=true, debug=false, algorithm=7, padding=false)
        # else
        #     println("$n $p $q nan")
        end
    end
end

