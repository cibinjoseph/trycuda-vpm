include("vpm.jl")
include("optimalpq.jl")

rectangular = false

# Read input arg if available
nt_default = 512
nt = length(ARGS) == 0 ? nt_default : parse(Int, ARGS[1])

ar = collect(10:10:2000)

for i in ar
    ns = Int32(nt * i)
    if rectangular
        p, q, r = optimal_pqr(ns; multiple32=true)
    else
        p, q, _ = optimal_pq(nt; multiple32=true)
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
