include("vpm.jl")

nt = 256
ns_list = 2 .^ (8:2:20)
algorithm_list = [7, 8, 9, 11]
p = 32
q = 4
output_file = "output_correctness.csv"  # set to nothing to skip writing

header = "algorithm nt ns v_L2 v_Linf j_L2 j_Linf"
println(header)

io = nothing
if output_file !== nothing
    io = open(output_file, "w")
    println(io, header)
end

for algorithm in algorithm_list
    for ns in ns_list
        errors = main(1; ns=ns, nt=nt, p=p, q=q, r=64, show_pq=false, debug=false,
                      algorithm=algorithm, padding=false, return_errors=true, verbose=false)
        v_L2 = errors.velocity.L2
        v_Linf = errors.velocity.Linf
        j_L2 = errors.gradient.L2
        j_Linf = errors.gradient.Linf
        row = "$algorithm $nt $ns $v_L2 $v_Linf $j_L2 $j_Linf"
        println(row)
        if io !== nothing
            println(io, row)
        end
    end
end

if io !== nothing
    close(io)
end
