using CUDA
using BenchmarkTools
using Random
using Statistics

const const4 = 1/(4*pi)

@inline g_val(r) = r^3 * (r^2 + 2.5) / (r^2 + 1)^2.5
@inline dg_val(r) = 7.5 * r^2 / ((r^2 + 1)^2.5*(r^2 + 1))
@inline cross_op(v) = [0.0 -v[3] v[2]; v[3] 0.0 -v[1]; -v[2] v[1] 0.0];

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
            r3 = r * r2

            g_sgm = g_val(r/ps[7])
            dg_sgmdr = dg_val(r/ps[7])

            crss1 = -const4 / r^3 * ( dX2*ps[6] - dX3*ps[5] )
            crss2 = -const4 / r^3 * ( dX3*ps[4] - dX1*ps[6] )
            crss3 = -const4 / r^3 * ( dX1*ps[5] - dX2*ps[4] )

            pt[10] += g_sgm * crss1
            pt[11] += g_sgm * crss2
            pt[12] += g_sgm * crss3

            aux = dg_sgmdr/(ps[7]*r) - 3*g_sgm /r^2

            pt[16] += aux * crss1 * dX1
            pt[17] += aux * crss2 * dX2
            pt[18] += aux * crss3 * dX3

            pt[19] += aux * crss1 * dX2
            pt[20] += aux * crss2 * dX2
            pt[21] += aux * crss3 * dX2

            pt[22] += aux * crss1 * dX3
            pt[23] += aux * crss2 * dX3
            pt[24] += aux * crss3 * dX3

            aux = -const4 * g_sgm/r3

            pt[17] -= aux * ps[6]
            pt[18] += aux * ps[5]

            pt[19] -= aux * ps[6]
            pt[21] += aux * ps[4]

            pt[22] -= aux * ps[5]
            pt[23] += aux * ps[4]
        end
    end
end

function gpu_interact(s, t)
    # Allocate required intermmediate arrays
    nt = size(t, 2)
    ns = size(s, 2)

    dX = CuArray{eltype(t)}(undef, (3, nt))
    crss = similar(dX)
    U = CUDA.zeros(size(dX))
    J1 = CUDA.zeros(size(dX))
    J2 = CUDA.zeros(size(dX))
    J3 = CUDA.zeros(size(dX))

    r = CuArray{eltype(t)}(undef, (1, nt))
    r2 = similar(r)
    r3 = similar(r)
    rbysigma = similar(r)
    g_sgm = similar(r)
    dg_sgm = similar(r)

    dX_cpu = Array{eltype(t)}(undef, (3, nt))

    s_sigma = view(s, 7, :)
    s_gamma_mat = CuArray{eltype(t)}(undef, (3, 3))

    # Loop over each source
    for i = 1:size(s, 2)
        @inbounds dX_cpu .= view(t, 1:3, :) .- view(s, 1:3, i)
        copyto!(dX, dX_cpu)
        r2 .= mapreduce(x->x^2, +, dX, dims=1)
        # r2 .= CUDA.sum(CUDA.abs2, dX, dims=1)
        r .= CUDA.sqrt.(r2)
        r3 = r .* r2

        @inbounds rbysigma .= r / s_sigma[i]
        g_sgm .= map(g_val, rbysigma)
        dg_sgm .= map(dg_val, rbysigma)

        @inbounds copyto!(s_gamma_mat, cross_op(view(s, 4:6, i)))
        crss .= const4 * (s_gamma_mat * dX) ./ r3

        U .+= g_sgm .* crss

        @inbounds aux = dg_sgm ./ (r * s_sigma[i]) .- 3*map(/, g_sgm, r2)
        dX .= aux .* dX

        @inbounds J1 .+= reshape(dX[1, :], size(r)) .* crss
        @inbounds J2 .+= reshape(dX[2, :], size(r)) .* crss
        @inbounds J3 .+= reshape(dX[3, :], size(r)) .* crss

        aux .= -const4 * map(/, g_sgm, r3)
        Jterm1 = aux .* view(s, 4, i)

    end
    t[10:12, :] .= Array(U)
    t[16:18, :] .+= Array(J1)
    t[19:21, :] .+= Array(J2)
    t[22:24, :] .+= Array(J3)
end

# ns = 2 .^ collect(4:2:20)
n = 2^14
nfields = 43

# for n in ns
src, trg, src2, trg2 = get_inputs(n, nfields)
t_cpu = @benchmark cpu_interact($src, $trg)
t_gpu = @benchmark CUDA.@sync gpu_interact($src, $trg)
speedup = mean(t_cpu.times)/mean(t_gpu.times)
println(n, " ", speedup)
# end
