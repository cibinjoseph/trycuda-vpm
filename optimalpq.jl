using StaticArrays
include("vpm.jl")


function get_pq(productmax=512, multiple=32)
    pq = Vector{Tuple{Int, Int}}()
    for p in 1:productmax, q in 1:productmax
        prod = p*q
        if prod % multiple == 0 && prod <= productmax && p >= q
            push!(pq, (p, q))
        end
    end
    return pq
end

function write_pq(filename, productmax=512, multiple=32)
    pq = get_pq(productmax, multiple)
    p = [t[1] for t in pq]
    q = [t[2] for t in pq]
    pq_mat = [p'; q']

    writedlm(filename, pq_mat, ' ')
end

function get_all_pq(n)
    divs = sort(partial_divisors(n; maxdiv=n))
    pq = Vector{Tuple{Int, Int}}()
    for p in divs, q in divs
        if p >= q && p*q <= 512
            push!(pq, (p, q))
        end
    end
    return pq
end

function max_speedup(n)
    pqs = get_all_pq(n)
    # pqs = get_pq()
    speedup_max = 1
    p_max = 1
    q_max = 1
    for (p, q) in pqs
        _, _, _, _, speedup = main(3; nt=n, ns=n, p=p, q=q, debug=false, algorithm=7, padding=false, show_pq=true, return_vals=true)
        if speedup >= speedup_max
            p_max, q_max = p, q
            speedup_max = speedup
        end
    end
    return p_max, q_max, speedup_max
end

nsamples = 20
nvals = rand(1000:100_000, nsamples)

# open("heuristic.csv", "w") do fh
#     println(fh, "n p_max q_max speedup_max popt qopt popt32 qopt32")

#     for (i, n) in enumerate(nvals)
#         println("Case $i $n")
#         p_max, q_max, speedup_max = max_speedup(n)
#         popt, qopt, _ = optimal_pq(n; multiple32=false)
#         popt32, qopt32, _ = optimal_pq(n; multiple32=true)
#         println(fh, "$n $p_max $q_max $speedup_max $popt $qopt $popt32 $qopt32")
#         flush(fh)
#     end
# end

# for n in nvals
#     p, q = optimal_pq(n; multiple32=false)
#     println("$n $p $q")
# end
# for i in vcat([1], 2:2:160)
#     p, q, npad = optimal_pq(i; multiple32=true)
#     println("$i. $(i*1250) $p $q $npad $(i%32)")
# end
return
