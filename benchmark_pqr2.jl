include("vpm.jl")
include("optimalpq.jl")


# Read input arg if available
nt_default = 512
option_default = 0
nt = length(ARGS) == 0 ? nt_default : parse(Int, ARGS[1])
option = length(ARGS) <= 1 ? option_default : parse(Int, ARGS[2])

# Rectangular if option >= 0
rectangular = option >= 0 ? true : false

multiple32 = true
if option == 1
    multiple32 = false
end

ar = collect(10:10:2000)

for i in ar
    ns = Int32(nt * i)
    if rectangular
        p, q, r = optimal_pqr(ns; multiple32=multiple32, pmax=1)
    else
        p, q, _ = optimal_pq(nt; multiple32=multiple32)
        r = p
    end

    p = Int32(p)
    q = Int32(q)
    r = Int32(r)

    if rectangular
        main(3; ns=ns, nt=nt, p=p, q=q, r=r, show_pq=true, debug=false, algorithm=9, padding=false, return_vals=false)
    else
        main(3; ns=ns, nt=nt, p=p, q=q, r=r, show_pq=true, debug=false, algorithm=7, padding=false, return_vals=false)
    end
end
