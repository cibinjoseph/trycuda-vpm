using StaticArrays


function get_score(pin, qin; pmax=512, qmax=8, α=0.0, β=0.0)
    p = log(pin)/log(pmax)
    q = qin/qmax
    α = abs(α) < 1e-6 ? 0.1 : α
    β = abs(β) < 1e-6 ? 0.5 : β
    score = -(p-(0.7-α*q))^2 + 1 + β*q
    return score
end

function partial_divisors(n; maxdiv=512)
    divs = Vector{typeof(n)}()
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
        @warn "This function is hardcoded for productmax=512.
        # multiple32=true will be ignored."
        popt, qopt = p, q
    end
    return popt, qopt
end

function optimal_pq(n; productmax=512, multiple32=true, pmax=512, qmax=8, α=0.0, β=0.0)
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

    qvals = copy(divs)
    filter!(x -> x <= qmax, qvals)

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
        popt, qopt = closest_tuple_32(popt, qopt; productmax=productmax)
    end

    ropt = 1 
    return popt, qopt, ropt
end

function optimal_pqr(ns; productmax=512, multiple32=true, pmax=1, qmax=512, rmax=876, α=0.0, β=0.0)
    popt = pmax
    divs = partial_divisors(ns; maxdiv=max(qmax, rmax))
    maxval = maximum(divs)
    ropt = min(rmax, maxval)
    qopt = 1
    for div in divs
        if div > qopt && popt*div <= productmax
            qopt = div
        end
    end

    if multiple32
        qopt = cld(popt*qopt, 32) * 32 ÷ popt
    end

    return popt, qopt, ropt
end

