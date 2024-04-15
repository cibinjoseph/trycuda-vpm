using CUDA
using BenchmarkTools
using Random
using Statistics


@inline g_val(r) = r^3 * (r^2 + 2.5) / (r^2 + 1)^2.5
@inline dg_val(r) = 7.5 * r^2 / ((r^2 + 1)^2.5*(r^2 + 1))

function cpu_interact(s, t)
    for ps in eachcol(view(s, :, :))
        for pt in eachcol(view(t, :, :))
            # Operation 1
            dX1 = ps[1] - pt[1]
            dX2 = ps[2] - pt[2]
            dX3 = ps[3] - pt[3]
        end
    end
end

function gpu_interact(s, t)
    # Allocate GPU arrays
    dX = similar(t[1:3, :])
    for ps in eachcol(view(s, :, :))
        interact_targets(ps, t, dX)
    end
end

function interact_targets(ps, ts, dX)
    # Operation 1
    @views dX .= ts[1:3, :] .- ps[1:3]
end

function get_inputs(nparticles, nfields; seed=1234, T=Float32)
    Random.seed!(seed)
    src = rand(T, nfields, nparticles)
    trg = rand(T, nfields, nparticles)

    src2 = deepcopy(src)
    trg2 = deepcopy(trg)
    return src, trg, src2, trg2
end

ns = 2 .^ collect(4:2:16)
nfields = 43

for n in ns
    src, trg, src2, trg2 = get_inputs(n, nfields)
    t_cpu = @benchmark cpu_interact($src, $trg)
    t_gpu = @benchmark gpu_interact($src, $trg)
    speedup = mean(t_cpu.times)/mean(t_gpu.times)
    # println(n, " ", mean(t_cpu.times), " ", mean(t_gpu.times), " ", speedup)
    println(n, " ", speedup)
end
