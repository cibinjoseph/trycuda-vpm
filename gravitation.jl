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

@inline function interaction!(t, s, i, j)
    @inbounds r_1 = s[1, j] - t[1, i]
    @inbounds r_2 = s[2, j] - t[2, i]
    @inbounds r_3 = s[3, j] - t[3, i]
    r_sqr = r_1*r_1 + r_2*r_2 + r_3*r_3 + eps2
    r_cube = r_sqr*r_sqr*r_sqr
    @inbounds mag = s[4, j] / sqrt(r_cube)

    @inbounds t[5, i] += r_1*mag
    @inbounds t[6, i] += r_2*mag
    @inbounds t[7, i] += r_3*mag
end

function cpu_gravity!(s, t)
    for i in 1:size(t, 2)
        for j in 1:size(s, 2)
            interaction!(t, s, i, j)
        end
    end
end

# Naive implementation
# Each thread handles a single target and uses global GPU memory
function gpu_gravity1!(s::CuDeviceMatrix{T}, t::CuDeviceMatrix{T}) where T
    idx::Int32 = threadIdx().x+(blockIdx().x-1)*blockDim().x

    t_size::Int32 = size(t, 2)
    s_size::Int32 = size(s, 2)

    i::Int32 = idx
    if i <= t_size
        j::Int32 = 1
        while j <= s_size
            interaction!(t, s, i, j)
            j += 1
        end
    end
    return
end

# Better implementation
# Each thread handles a single target and uses local GPU memory
function gpu_gravity2!(s::CuDeviceMatrix{T}, t::CuDeviceMatrix{T}) where T
    ithread::Int32 = threadIdx().x
    tile_dim::Int32 = blockDim().x
    itarget::Int32 = ithread+(blockIdx().x-1)*blockDim().x

    t_size::Int32 = size(t, 2)
    s_size::Int32 = size(s, 2)

    n_tiles::Int32 = t_size/tile_dim

    sh_mem = CuDynamicSharedArray(T, (4, tile_dim))

    itile::Int32 = 1
    while itile <= n_tiles
        # Each thread will copy source coordinates corresponding to its index into shared memory
        @inbounds sh_mem[1, ithread] = s[1, ithread + (itile-1)*tile_dim]
        @inbounds sh_mem[2, ithread] = s[2, ithread + (itile-1)*tile_dim]
        @inbounds sh_mem[3, ithread] = s[3, ithread + (itile-1)*tile_dim]
        @inbounds sh_mem[4, ithread] = s[4, ithread + (itile-1)*tile_dim]
        sync_threads()

        # Each thread will compute the influence of all the sources in the shared memory on the target corresponding to its index
        isource::Int32 = 1
        while isource <= tile_dim
            interaction!(t, sh_mem, itarget, isource)
            isource += 1
        end
        itile += 1
        sync_threads()
    end
    return
end

# Each thread handles a single target and uses local GPU memory
# Multiple block dimensions for spatial dimensions
function gpu_gravity3!(s::CuDeviceMatrix{T}, t::CuDeviceMatrix{T}) where T
    ithread::Int32 = threadIdx().x
    tile_dim::Int32 = blockDim().x
    itarget::Int32 = ithread+(blockIdx().x-1)*blockDim().x

    t_size::Int32 = size(t, 2)
    s_size::Int32 = size(s, 2)

    n_tiles::Int32 = t_size/tile_dim

    sh_mem = CuDynamicSharedArray(T, (4, tile_dim))

    itile::Int32 = 1
    while itile <= n_tiles
        # Each thread will copy source coordinates corresponding to its index into shared memory
        @inbounds sh_mem[1, ithread] = s[1, ithread + (itile-1)*tile_dim]
        @inbounds sh_mem[2, ithread] = s[2, ithread + (itile-1)*tile_dim]
        @inbounds sh_mem[3, ithread] = s[3, ithread + (itile-1)*tile_dim]
        @inbounds sh_mem[4, ithread] = s[4, ithread + (itile-1)*tile_dim]
        sync_threads()

        # Each thread will compute the influence of all the sources in the shared memory on the target corresponding to its index
        isource::Int32 = 1
        while isource <= tile_dim
            interaction!(t, sh_mem, itarget, isource)
            isource+= 1
        end
        itile += 1
        sync_threads()
    end
    return
end

function benchmark1_gpu!(s, t)
    s_d = CuArray(view(s, 1:4, :))
    t_d = CuArray(t)

    kernel = @cuda launch=false gpu_gravity1!(s_d, t_d)
    config = launch_configuration(kernel.fun)
    threads = min(size(t, 2), config.threads)
    blocks = cld(size(t, 2), threads)

    CUDA.@sync kernel(s_d, t_d; threads, blocks)

    view(t, 5:7, :) .= Array(t_d[end-2:end, :])
end

function benchmark2_gpu!(s, t, p)
    s_d = CuArray(view(s, 1:4, :))
    t_d = CuArray(t)

    # Num of threads in a tile should always be 
    # less than number of threads in a block (1024)
    # or limited by memory size
    threads = p
    blocks = cld(size(s, 2), p)
    shmem = sizeof(zeros(Float32, 4, p))
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks shmem=shmem gpu_gravity2!(s_d, t_d)
    end

    view(t, 5:7, :) .= Array(t_d[end-2:end, :])
end

function benchmark3_gpu!(s, t, p)
    s_d = CuArray(view(s, 1:4, :))
    t_d = CuArray(t)

    # Num of threads in a tile should always be 
    # less than number of threads in a block (1024)
    # or limited by memory size
    threads = p
    blocks = cld(size(s, 2), p)
    shmem = sizeof(zeros(Float32, 4, p))
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks shmem=shmem gpu_gravity3!(s_d, t_d)
    end

    view(t, 5:7, :) .= Array(t_d[end-2:end, :])
end

function main(run_option)
    nfields = 7
    if run_option == 1 || run_option == 2
        nparticles = 2^17
        println("No. of particles: $nparticles")
        p = min(2^10, nparticles, 1024)
        println("Tile size: $p")
        src, trg, src2, trg2 = get_inputs(nparticles, nfields)
        # cpu_gravity!(src, trg)
        if run_option == 1
            benchmark2_gpu!(src2, trg2, p)
            diff = abs.(trg .- trg2) .< Float32(1E-4)
            if all(diff)
                println("MATCHES")
            else
                println("DOES NOT MATCH!")
                # display(diff)
            end
        else
            println("Running profiler...")
            CUDA.@profile external=true benchmark2_gpu!(src2, trg2, p)
        end
    else
        ns = 2 .^ collect(4:1:17)
        for nparticles in ns
            p = min(2^10, nparticles, 1024)
            println("Tile size: $p")
            src, trg, src2, trg2 = get_inputs(nparticles, nfields)
            t_cpu = @benchmark cpu_gravity!($src, $trg)
            t_gpu = @benchmark benchmark2_gpu!($src2, $trg2, $p)
            speedup = median(t_cpu.times)/median(t_gpu.times)
            println("$nparticles $speedup")
        end
    end
    return
end

# Run_option - # [1]test [2]profile [3]benchmark
run_option = 2
main(run_option)
