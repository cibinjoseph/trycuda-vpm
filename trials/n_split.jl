function common_n(st)
    n_ret = 0
    st_ret = st
    if st[1] < st[2]
        n_ret, st_ret = st[1], (st[1], st[2]-st[1])
    elseif st[1] > st[2]
        n_ret, st_ret = st[2], (st[1]-st[2], st[2])
    end
    return n_ret, st_ret
end

function recurse_commons(st; until_n=1)
    i = 1
    st_split = st
    splits = [st]
    while true
        i, st_split = common_n(st_split)
        push!(splits, st_split)
        if min(st_split...) <= until_n || i == 0
            break
        end
    end
    return splits
end

function split_indices(st; until_n=1)
    splits = recurse_commons(st; until_n=until_n)
    s_start, t_start = 1, 1
    s_end, t_end = splits[1][1], splits[1][2]

    for i in 2:length(splits)
        s_used = splits[i-1][1] - splits[i][1]
        t_used = splits[i-1][2] - splits[i][2]

        if s_used > 0
            println("S=$s_start:$(s_start+s_used-1) T=$t_start:$t_end")
            s_start += s_used
        elseif t_used > 0
            println("S=$s_start:$s_end T=$t_start:$(t_start+t_used-1)")
            t_start += t_used
        end
    end

end
