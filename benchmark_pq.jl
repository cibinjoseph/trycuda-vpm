include("vpm.jl")
using Primes

ns = 5000
divs = sort(divisors(ns))


for p in divs
    divs_p = divisors(p)
    for q in divs
        if q in divs_p && p*q <= 512
            main(3; ns=ns, p=p, q=q, show_pq=true, debug=false, algorithm=7, padding=false)
        else
            println("$ns $p $q nan")
        end
    end
end

