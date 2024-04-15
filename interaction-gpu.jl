using CUDA
using BenchmarkTools
using Random
using Statistics


@inline g_val(r) = r^3 * (r^2 + 2.5) / (r^2 + 1)^2.5
@inline dg_val(r) = 7.5 * r^2 / ((r^2 + 1)^2.5*(r^2 + 1))

function get_inputs(nparticles, nfields; seed=1234, T=Float32)
    Random.seed!(seed)
    src = rand(T, nfields, nparticles)
    trg = rand(T, nfields, nparticles)

    src2 = deepcopy(src)
    trg2 = deepcopy(trg)
    return src, trg, src2, trg2
end

function cpu_interact(s, t)
    for ps in eachcol(view(s, :, :))
        for pt in eachcol(view(t, :, :))
            # Operation 1
            dX1 = ps[1] - pt[1]
            dX2 = ps[2] - pt[2]
            dX3 = ps[3] - pt[3]

            # Operation 2
            r2 = dX1*dX1 + dX2*dX2 + dX3*dX3
            r = sqrt(r2)
        end
    end
end

function gpu_interact(s, t)
    # Allocate main GPU arrays
    sx_d = CuArray(view(s, 1:3, :))
    # t_d = CuArray(view(t, :, :))
    tx_d = CuArray(view(t, 1:3, :))

    # Allocate required intermmediate arrays
    nt = size(t, 2)
    dX = CuArray{eltype(t)}(undef, (3, nt))
    r2 = CuArray{eltype(t)}(undef, (1, nt))
    r = CuArray{eltype(t)}(undef, (1, nt))

    dX_cpu = Array{eltype(t)}(undef, (3, nt))

    for i = 1:size(s, 2)
        # Operation 1
        dX_cpu .= view(t, 1:3, :) .- view(s, 1:3, i)
        copyto!(dX, dX_cpu)
        # Operation 2
        r2 .= mapreduce(x->x^2, +, dX, dims=1)
        # r2 .= CUDA.sum(CUDA.abs2, dX, dims=1)
        r .= CUDA.sqrt.(r2)
    end
end

# ns = 2 .^ collect(4:2:20)
n = 2^12
nfields = 43

# for n in ns
    src, trg, src2, trg2 = get_inputs(n, nfields)
    t_cpu = @benchmark cpu_interact($src, $trg)
    t_gpu = @benchmark CUDA.@sync gpu_interact($src, $trg)
    speedup = mean(t_cpu.times)/mean(t_gpu.times)
    println(n, " ", speedup)
# end
