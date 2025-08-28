function extract_max_speedups(file_path::String)
    max_data = Dict{Int, NamedTuple{(:speedup, :p, :q, :r), Tuple{Float64, Int, Int, Int}}}()
    current_n = 0

    for line in eachline(file_path)
        if startswith(line, "Case")
            # Extract the n value
            _, _, n_str = split(line)
            current_n = parse(Int, n_str)
            max_data[current_n] = (speedup = -Inf, p = 0, q = 0, r = 0)
        else
            values = split(line)
            n = parse(Int, values[1])
            p = parse(Int, values[2])
            q = parse(Int, values[3])
            r = parse(Int, values[4])
            speedup = parse(Float64, values[5])

            if speedup > max_data[n].speedup
                max_data[n] = (speedup = speedup, p = p, q = q, r = r)
            end
        end
    end

    return max_data
end

sp = extract_max_speedups("max2.log")
include("optimalpq.jl")
for n in keys(sp)
    popt, qopt = optimal_pq(n)
end

open("maxspeedup.csv", "w") do fh
    for n in keys(sp)
        if n%2 == 0
            println(fh, "$n $(sp[n].speedup)")
        end
    end
end
