using CUDA
using BenchmarkTools
using Random
using Statistics

const eps2 = 1e-6

function get_inputs(nparticles, nfields; seed=1234, T=Float32)
    Random.seed!(seed)
    src = rand(T, nfields, nparticles)
    trg = rand(T, nfields, nparticles)

    src2 = deepcopy(src)
    trg2 = deepcopy(trg)
    return src, trg, src2, trg2
end

function cpu_gravity!(s::Matrix{T}, t::Matrix{T}) where T
    for i in 1:size(t, 2)
        for j in 1:size(s, 2)
            r_1 = s[1, j] - t[1, i]
            r_2 = s[2, j] - t[2, i]
            r_3 = s[3, j] - t[3, i]
            r_sqr = r_1*r_1 + r_2*r_2 + r_3*r_3 + eps2
            r_cube = r_sqr*r_sqr*r_sqr
            mag = s[4, j] / sqrt(r_cube)

            t[5, i] += r_1*mag
            t[6, i] += r_2*mag
            t[7, i] += r_3*mag
        end
    end
end

function gpu_gravity1!(s::CuDeviceMatrix{T}, t::CuDeviceMatrix{T}) where T
    idx::Int32 = threadIdx().x+blockIdx().x*(blockDim().x-1)
    stride::Int32 = gridDim().x * blockDim().x

    t_size::Int32 = size(t, 2)
    s_size::Int32 = size(s, 2)

    i::Int32 = idx
    while i <= t_size
        j::Int32 = 1
        while j <= s_size
            @inbounds r_1 = s[1, j] - t[1, i]
            @inbounds r_2 = s[2, j] - t[2, i]
            @inbounds r_3 = s[3, j] - t[3, i]
            r_sqr = r_1*r_1 + r_2*r_2 + r_3*r_3 + eps2
            r_cube = r_sqr*r_sqr*r_sqr
            @inbounds mag = s[4, j] / sqrt(r_cube)

            @inbounds t[5, i] += r_1*mag
            @inbounds t[6, i] += r_2*mag
            @inbounds t[7, i] += r_3*mag
            j += 1
        end
        i += stride
    end
    return
end

function benchmark_gpu!(s, t)
    s_d = CuArray(s)
    t_d = CuArray(t)

    kernel = @cuda launch=false gpu_gravity1!(s_d, t_d)
    config = launch_configuration(kernel.fun)
    threads = min(size(t, 2), config.threads)
    blocks = cld(size(t, 2), threads)
    @show threads, blocks

    CUDA.@sync kernel(s_d, t_d; threads, blocks)

    t = Array(t_d)
end

ns = 2 .^ collect(1:1:17)
nfields = 7
for nparticles in ns
    src, trg, src2, trg2 = get_inputs(nparticles, nfields)
    t_cpu = @benchmark cpu_gravity!($src, $trg)
    t_gpu = @benchmark benchmark_gpu!($src2, $trg2)
    speedup = median(t_cpu.times)/median(t_gpu.times)
    println("$nparticles $speedup")
end