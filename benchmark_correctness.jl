include("vpm.jl")

nt = 256
ns_list = 2 .^ (8:2:20)

p= 32
q = 2

println("nt ns v_L2 v_Linf j_L2 j_Linf")
for ns in ns_list
    errors = main(1; ns=ns, nt=nt, p=p, q=q, r=4, show_pq=false, debug=false, algorithm=11, padding=false, return_errors=true, verbose=false)

    v_L2 = errors.velocity.L2
    v_Linf = errors.velocity.Linf
    j_L2 = errors.gradient.L2
    j_Linf = errors.gradient.Linf
    println("$nt $ns $v_L2 $v_Linf $j_L2 $j_Linf")
end
