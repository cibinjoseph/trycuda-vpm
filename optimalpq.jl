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

function get_score(pin, qin; pmax=512, qmax=8, α=0.0, β=0.0)
    p = log(pin)/log(pmax)
    q = qin/qmax
    α = abs(α) < 1e-6 ? 0.1 : α
    β = abs(β) < 1e-6 ? 0.5 : β
    # score = -(p-(0.7-α*q))^2 + (1-(0.5-β*q))^2
    score = -(p-(0.7-α*q))^2 + 1 + β*q
    return score
end

function partial_divisors(n::Int; maxdiv::Int=512)
    divs = Int[]
    sqrt_n = isqrt(n)
    for i in 1:min(maxdiv, sqrt_n)
        if n % i == 0
            push!(divs, i)
            q = div(n, i)
            if i != q && q <= maxdiv
                push!(divs, q)
            end
        end
    end
    return sort!(divs)
end

include("divs_512.jl")
function closest_tuple_32(p, q; productmax=512, dist_threshold=10)
    if productmax == 512
        dist_min = 1000
        popt = p
        qopt = q
        for i in 1:size(divs_512, 2)
            dist = (divs_512[1, i]-p)^2 + (divs_512[2, i]-q)^2
            if dist <= dist_threshold && dist <= dist_min
                dist_min = dist
                popt = divs_512[1, i]
                qopt = divs_512[2, i]
            end
        end
    else
        @error "This function is hardcoded for productmax=512"
    end
    return popt, qopt
end

function optimal_pq(n; productmax=512, multiple32=false, pmax=512, qmax=8, α=0.0, β=0.0)
    divs = partial_divisors(n; maxdiv=productmax)
    npad = 0

    for k=1:10
        if length(divs) >= 8
            break
        else
            npad = k
            divs = partial_divisors(n+k; maxdiv=productmax)
        end
    end

    # pvals = copy(divs)
    # filter!(x -> x < pmax, pvals)
    # sort!(pvals)

    qvals = copy(divs)
    filter!(x -> x <= qmax, qvals)
    # sort!(qvals)

    popt = 1
    qopt = 1
    scoremax = 0
    for p in divs, q in qvals
        prod = p*q
        if prod <= productmax && p >= q
            score = get_score(p, q; pmax=pmax, qmax=qmax, α=α, β=β)
            if score >= scoremax
                popt = p
                qopt = q
                scoremax = score
            end
        end
    end

    # Find closest pair multiple of 32
    if multiple32
        popt, qopt = closest_tuple_32(popt, qopt; productmax=512)
    end
    return popt, qopt, npad
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
