function check_launch(n, p, q, productmax=default_productmax; throw_error=false)
    if p > n; throw_error && error("p must be less than or equal to n"); return false; end
    if p*q > productmax; throw_error && error("p*q must be less than $productmax"); return false; end
    if q > p; throw_error && error("q must be less than or equal to p"); return false; end
    if n % p != 0; throw_error && error("n must be divisible by p"); return false; end
    if p % q != 0; throw_error && error("p must be divisible by q"); return false; end

    return true
end

function check_launch(nt, ns, p, q, r, productmax=default_productmax; throw_error=false)
    if p > nt; throw_error && error("p must be less than or equal to nt"); return false; end
    if p*q > productmax; throw_error && error("p*q must be less than $productmax"); return false; end
    # if q > p; throw_error && error("q must be less than or equal to p"); return false; end
    if q > r; throw_error && error("q must be less than or equal to r"); return false; end
    if nt % p != 0; throw_error && error("nt must be divisible by p"); return false; end
    # if p % q != 0; throw_error && error("p must be divisible by q"); return false; end
    if ns % r != 0; throw_error && error("ns must be divisible by p"); return false; end
    if r % q != 0; throw_error && error("r must be divisible by q"); return false; end

    return true
end


function get_launch_config(nt; p_max=0, q_max=0, max_threads_per_block=512)
    p_max = (p_max == 0) ? max_threads_per_block : p_max
    q_max = (q_max == 0) ? p_max : q_max

    divs_n = sort(divisors(Int32(nt)))
    p = 1
    q = 1
    ip = 1
    for (i, div) in enumerate(divs_n)
        if div <= p_max
            p = div
            ip = i
        else
            break
        end
    end

    # Decision algorithm 1: Creates a matrix using indices and finds max of
    # weighted sum of indices

    i_weight = 0
    j_weight = 1-i_weight

    max_ij = i_weight*ip + j_weight*1
    isgood = true
    for i in 1:ip
        for j in 1:ip
            isgood = check_launch(nt, divs_n[i], divs_n[j], max_threads_per_block)
            if isgood && (divs_n[i] <= p_max)
                # Check if this is the max achievable ij value
                # in the p, q choice matrix
                obj_val = i_weight*i+j_weight*j
                if (obj_val >= max_ij) && (divs_n[j] <= q_max)
                    max_ij = obj_val
                    p = divs_n[i]
                    q = divs_n[j]
                end
            end
        end
    end

    return p, q
end

function get_launch_config(nt, ns; p_max=0, q_max=0, r_max=875, max_threads_per_block=512)
    # r_max=875 corresponds to 48KB in shared memory
    p_max = (p_max == 0) ? max_threads_per_block : p_max
    q_max = (q_max == 0) ? max_threads_per_block : q_max

    # Find p
    divs_nt = sort(divisors(Int32(nt)))
    p = 1
    q = 1
    ip = 1
    for (i, div) in enumerate(divs_nt)
        if div <= p_max
            p = div
            ip = i
        else
            break
        end
    end

    # Find r
    divs_ns = sort(divisors(Int32(ns)))
    r = 1
    ir = 1
    for (i, div) in enumerate(divs_ns)
        if div <= r_max
            r = div
            ir = i
        else
            break
        end
    end

    # Decision algorithm 1: Creates a matrix using indices and finds max of
    # weighted sum of indices

    # Find q based on r
    i_weight = 0
    j_weight = 1-i_weight

    max_ij = i_weight*ip + j_weight*1
    isgood = true
    for i in 1:ip
        for j in 1:ir
            isgood = check_launch(nt, ns, divs_nt[i], divs_ns[j], r, max_threads_per_block)
            # isgood = divs_nt[i]*divs_ns[j] < max_threads_per_block
            if isgood && (divs_nt[i] <= p_max)
                # Check if this is the max achievable ij value
                # in the p, q choice matrix
                obj_val = i_weight*i+j_weight*j
                if (obj_val >= max_ij) && (divs_ns[j] <= q_max)
                    max_ij = obj_val
                    p = divs_nt[i]
                    q = divs_ns[j]
                end
            end
        end
    end

    return p, q, r
end

function get_launch_config1(nt, ns; p_max=0, q_max=0, r_max=875, max_threads_per_block=512)
    # r_max=875 corresponds to 48KB in shared memory
    p_max = (p_max == 0) ? max_threads_per_block : p_max
    q_max = (q_max == 0) ? max_threads_per_block : q_max

    # Find possible divisors
    divs_nt = sort(divisors(Int32(nt)))
    divs_ns = sort(divisors(Int32(ns)))

    # Reduce divisors search space
    divs_nt = divs_nt[findall(divs_nt .<= max_threads_per_block)]
    divs_nt = p_max > 0 ? divs_nt[findall(divs_nt .<= p_max)] : divs_nt
    divs_ns = divs_ns[findall(divs_ns .<= max_threads_per_block)]
    divs_ns = q_max > 0 ? divs_ns[findall(divs_ns .<= q_max)] : divs_ns
    divs_ns = r_max > 0 ? divs_ns[findall(divs_ns .<= r_max)] : divs_ns

    p_weight = 0.5
    q_weight = 0.2
    r_weight = 0.3

    p, q, r = Int32(1), Int32(1), Int32(1)
    max_val = 0.0
    for (ir, rval) in enumerate(divs_ns),
        (iq, qval) in enumerate(divs_ns),
        (ip, pval) in enumerate(divs_nt)

        isgood = check_launch(nt, ns, pval, qval, rval, max_threads_per_block)
        if isgood && (pval<=p_max) && (qval<=q_max) && (rval<=r_max) && (pval>1) && (qval>1)
            # Check if this is the max achievable ij value
            # in the p, q choice matrix
            obj_val = p_weight*ip + q_weight*iq + r_weight*ir
            if obj_val >= max_val
                max_val = obj_val
                p = pval
                q = qval
                r = rval
            end
        end
    end

    return p, q, r
end

