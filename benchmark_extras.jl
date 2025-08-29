using DelimitedFiles


# 1) Create file with header if missing/empty
function ensure_header(path::AbstractString, header="")
    if !isfile(path) || filesize(path) == 0
        open(path, "w") do io
            println(io, header)
        end
    end
end

# 2) Re-scan the file to see if a case exists (re-reads ENTIRE file each call)
function case_exists(path::AbstractString, ncoeffs::Int, nparticles::Int)::Bool
    if isfile(path) && filesize(path) > 0
        open(path, "r") do io
            first = true
            for line in eachline(io)
                if first; first = false; continue; end  # skip header
                s = split(strip(line))
                length(s) < 2 && continue
                n1 = tryparse(Int, s[1]); n2 = tryparse(Int, s[2])
                if n1 !== nothing && n2 !== nothing && n1 == ncoeffs && n2 == nparticles
                    return true
                end
            end
        end
    end
    return false
end

# 3) Append one result row immediately
function append_row(path::AbstractString, row)
    open(path, "a") do io
        println(io, join(row..., ' '))
        flush(io)  # write-through so a crash preserves finished rows
    end
end
