using CUDA
using BenchmarkTools
using Random
using Statistics

const eps2 = 1e-6

function get_inputs(nparticles, nfields; T=Float32)
    Random.seed!(1234)  # This has to be present inside this function
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

@inline function gpu_interaction!(t, s, i, j)
    acc1, acc2, acc3 = 0.0f0, 0.0f0, 0.0f0
    @inbounds r_1 = s[1, j] - t[1, i]
    @inbounds r_2 = s[2, j] - t[2, i]
    @inbounds r_3 = s[3, j] - t[3, i]
    r_sqr = r_1*r_1 + r_2*r_2 + r_3*r_3 + eps2
    r_cube = r_sqr*r_sqr*r_sqr
    @inbounds mag = s[4, j] / sqrt(r_cube)

    # Add influence of a source
    acc1 += r_1*mag
    acc2 += r_2*mag
    acc3 += r_3*mag
    return acc1, acc2, acc3
end

function cpu_gravity!(s, t)
    for i in 1:size(t, 2)
        for j in 1:size(s, 2)
            interaction!(t, s, i, j)
        end
    end
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
function gpu_gravity1!(s, t)
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
function gpu_gravity2!(s, t)
    ithread::Int32 = threadIdx().x
    tile_dim::Int32 = blockDim().x
    itarget::Int32 = ithread+(blockIdx().x-1)*blockDim().x

    t_size::Int32 = size(t, 2)
    s_size::Int32 = size(s, 2)

    n_tiles::Int32 = t_size/tile_dim

    sh_mem = CuDynamicSharedArray(Float32, (4, tile_dim))

    acc1 = zero(eltype(s))
    acc2 = zero(eltype(s))
    acc3 = zero(eltype(s))

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
            out = gpu_interaction!(t, sh_mem, itarget, isource)

            # Sum up accelerations for each source in a tile
            acc1 += out[1]
            acc2 += out[2]
            acc3 += out[3]
            isource += 1
        end
        itile += 1
        sync_threads()
    end

    # Sum up accelerations for each target/thread
    t[5, itarget] += acc1
    t[6, itarget] += acc2
    t[7, itarget] += acc3
    return
end

# Each thread handles a single target and uses local GPU memory
function gpu_gravity3!(s, t, num_cols)
    t_size::Int32 = size(t, 2)
    s_size::Int32 = size(s, 2)

    ithread::Int32 = threadIdx().x
    tile_dim::Int32 = t_size/gridDim().x

    # Row and column indices of threads in a block
    row = (ithread-1) % tile_dim + 1
    col = floor(Int32, (ithread-1)/tile_dim) + 1

    itarget::Int32 = row + (blockIdx().x-1)*tile_dim

    n_tiles::Int32 = t_size/tile_dim
    bodies_per_col::Int32 = tile_dim / num_cols

    sh_mem = CuDynamicSharedArray(Float32, (4, tile_dim))

    acc1 = zero(eltype(s))
    acc2 = zero(eltype(s))
    acc3 = zero(eltype(s))

    itile::Int32 = 1
    while itile <= n_tiles
        # Each thread will copy source coordinates corresponding to its index into shared memory. This will be done for each tile.
        if (col == 1)
            idx::Int32 = row + (itile-1)*tile_dim
            @inbounds sh_mem[1, row] = s[1, idx]
            @inbounds sh_mem[2, row] = s[2, idx]
            @inbounds sh_mem[3, row] = s[3, idx]
            @inbounds sh_mem[4, row] = s[4, idx]
        end
        sync_threads()

        # Each thread will compute the influence of all the sources in the shared memory on the target corresponding to its index
        i::Int32 = 1
        while i <= bodies_per_col
            i_source::Int32 = i + bodies_per_col*(col-1)
            out = gpu_interaction!(t, sh_mem, itarget, i_source)
            @cushow i_source, out[1], out[2], out[3]

            # Sum up accelerations for each source in a tile
            acc1 += out[1]
            acc2 += out[2]
            acc3 += out[3]
            i += 1
        end
        itile += 1
        sync_threads()
    end

    # Sum up accelerations for each target/thread
    t[5, itarget] += acc1
    t[6, itarget] += acc2
    t[7, itarget] += acc3
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
    shmem = sizeof(Float32) * 4 * p
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks shmem=shmem gpu_gravity2!(s_d, t_d)
    end

    view(t, 5:7, :) .= Array(t_d[end-2:end, :])
end

function benchmark3_gpu!(s, t, p, q)
    s_d = CuArray(view(s, 1:4, :))
    t_d = CuArray(t)

    # Num of threads in a tile should always be 
    # less than number of threads in a block (1024)
    # or limited by memory size
    threads::Int32 = p*q
    blocks::Int32 = cld(size(s, 2), p)
    shmem = sizeof(Float32) * 4 * p
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks shmem=shmem gpu_gravity3!(s_d, t_d, q)
    end

    view(t, 5:7, :) .= Array(t_d[end-2:end, :])
end

function main(run_option)
    nfields = 7
    if run_option == 1 || run_option == 2
        nparticles = 2^2
        println("No. of particles: $nparticles")
        p = min(2, nparticles, 1024)
        q = 2
        println("Tile size: $p")
        println("Cols per tile: $q")
        src, trg, src2, trg2 = get_inputs(nparticles, nfields)
        if run_option == 1
            cpu_gravity!(src, trg)
            benchmark3_gpu!(src2, trg2, p, q)
            diff = abs.(trg .- trg2)
            err_norm = sqrt(sum(abs2, diff))
            diff_bool = diff .< Float32(1E-4)
            if all(diff_bool)
                println("MATCHES")
            else
                if nparticles < 10
                    # display(trg)
                    # display(trg2)
                    # display(diff)
                end
                n_diff = count(==(false), diff_bool)
                n_total = 3*size(trg, 2)
                println("$n_diff of $n_total elements DO NOT MATCH")
                println("Error norm: $err_norm")
            end
        else
            println("Running profiler...")
            CUDA.@profile external=true benchmark3_gpu!(src2, trg2, p)
        end
    else
        ns = 2 .^ collect(4:1:17)
        for nparticles in ns
            p = min(2^9, nparticles, 1024)
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
run_option = 1
main(run_option)
