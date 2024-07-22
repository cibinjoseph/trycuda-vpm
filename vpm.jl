using CUDA
using BenchmarkTools
using Random
using StaticArrays
using Primes

const eps2 = 1e-6
const const4 = 0.25/pi
const nfields = 43

# Definitions for GPU erf() function
include("my_erf.jl")

function get_inputs(ns, nfields; nt=0, T=Float32)
    Random.seed!(1234)  # This has to be present inside this function

    nt = nt==0 ? ns : nt

    src = rand(T, nfields, ns)
    trg = rand(T, nfields, nt)
    # Zero initial target UJ
    trg[10:12, :] .= zero(T)
    trg[16:24, :] .= zero(T)

    src2 = deepcopy(src)
    trg2 = deepcopy(trg)
    return src, trg, src2, trg2
end

@inline g_val(r) = r^3 * (r^2 + 2.5) / (r^2 + 1)^2.5
@inline dg_val(r) = 7.5 * r^2 / ((r^2 + 1)^2.5*(r^2 + 1))

@inline function interaction!(t::Array{T}, s::Array{T}, i, j) where T
    @inbounds dX1 = t[1, i] - s[1, j]
    @inbounds dX2 = t[2, i] - s[2, j]
    @inbounds dX3 = t[3, i] - s[3, j]
    r2 = dX1*dX1 + dX2*dX2 + dX3*dX3
    r = sqrt(r2)
    r3 = r*r2

    # Mapping to variables
    @inbounds gam1 = s[4, j]
    @inbounds gam2 = s[5, j]
    @inbounds gam3 = s[6, j]
    @inbounds sigma = s[7, j]

    if r2 > T(eps2) && abs(sigma) > T(eps2)
        # Regularizing function and deriv
        # g_sgm = g_val(r/sigma)
        # dg_sgmdr = dg_val(r/sigma)
        g_sgm, dg_sgmdr = cpu_g_dgdr(r/sigma)

        # K × Γp
        @inbounds crss1 = -T(const4) / r3 * ( dX2*gam3 - dX3*gam2 )
        @inbounds crss2 = -T(const4) / r3 * ( dX3*gam1 - dX1*gam3 )
        @inbounds crss3 = -T(const4) / r3 * ( dX1*gam2 - dX2*gam1 )

        # U = ∑g_σ(x-xp) * K(x-xp) × Γp
        @inbounds t[10, i] += g_sgm * crss1
        @inbounds t[11, i] += g_sgm * crss2
        @inbounds t[12, i] += g_sgm * crss3

        # ∂u∂xj(x) = ∑[ ∂gσ∂xj(x−xp) * K(x−xp)×Γp + gσ(x−xp) * ∂K∂xj(x−xp)×Γp ]
        # ∂u∂xj(x) = ∑p[(Δxj∂gσ∂r/(σr) − 3Δxjgσ/r^2) K(Δx)×Γp
        aux = dg_sgmdr/(sigma*r) - 3*g_sgm /r2
        # ∂u∂xj(x) = −∑gσ/(4πr^3) δij×Γp
        # Adds the Kronecker delta term
        aux2 = -T(const4) * g_sgm / r3
        # j=1
        @inbounds t[16, i] += aux * crss1 * dX1
        @inbounds t[17, i] += aux * crss2 * dX1 - aux2 * gam3
        @inbounds t[18, i] += aux * crss3 * dX1 + aux2 * gam2
        # j=2
        @inbounds t[19, i] += aux * crss1 * dX2 + aux2 * gam3
        @inbounds t[20, i] += aux * crss2 * dX2
        @inbounds t[21, i] += aux * crss3 * dX2 - aux2 * gam1
        # j=3
        @inbounds t[22, i] += aux * crss1 * dX3 - aux2 * gam2
        @inbounds t[23, i] += aux * crss2 * dX3 + aux2 * gam1
        @inbounds t[24, i] += aux * crss3 * dX3
    end
end

@inline function gpu_interaction(tx, ty, tz, s, j, kernel)
    T = eltype(s)
    @inbounds dX1 = tx - s[1, j]
    @inbounds dX2 = ty - s[2, j]
    @inbounds dX3 = tz - s[3, j]
    r2 = dX1*dX1 + dX2*dX2 + dX3*dX3
    r = sqrt(r2)
    r3 = r*r2

    # Mapping to variables
    @inbounds gam1 = s[4, j]
    @inbounds gam2 = s[5, j]
    @inbounds gam3 = s[6, j]
    @inbounds sigma = s[7, j]

    UJ = @MVector zeros(T, 12)

    if r2 > T(eps2) && abs(sigma) > T(eps2)
        # Regularizing function and deriv
        # g_sgm = g_val(r/sigma)
        # dg_sgmdr = dg_val(r/sigma)
        g_sgm, dg_sgmdr = kernel(r/sigma)

        # K × Γp
        crss1 = -T(const4) / r3 * ( dX2*gam3 - dX3*gam2 )
        crss2 = -T(const4) / r3 * ( dX3*gam1 - dX1*gam3 )
        crss3 = -T(const4) / r3 * ( dX1*gam2 - dX2*gam1 )

        # U = ∑g_σ(x-xp) * K(x-xp) × Γp
        UJ[1] = g_sgm * crss1
        UJ[2] = g_sgm * crss2
        UJ[3] = g_sgm * crss3

        # ∂u∂xj(x) = ∑[ ∂gσ∂xj(x−xp) * K(x−xp)×Γp + gσ(x−xp) * ∂K∂xj(x−xp)×Γp ]
        # ∂u∂xj(x) = ∑p[(Δxj∂gσ∂r/(σr) − 3Δxjgσ/r^2) K(Δx)×Γp
        aux = dg_sgmdr/(sigma*r) - 3*g_sgm /r2
        # ∂u∂xj(x) = −∑gσ/(4πr^3) δij×Γp
        # Adds the Kronecker delta term
        aux2 = -T(const4) * g_sgm / r3
        # j=1
        UJ[4] = aux * crss1 * dX1
        UJ[5] = aux * crss2 * dX1 - aux2 * gam3
        UJ[6] = aux * crss3 * dX1 + aux2 * gam2
        # j=2
        UJ[7] = aux * crss1 * dX2 + aux2 * gam3
        UJ[8] = aux * crss2 * dX2
        UJ[9] = aux * crss3 * dX2 - aux2 * gam1
        # j=3
        UJ[10] = aux * crss1 * dX3 - aux2 * gam2
        UJ[11] = aux * crss2 * dX3 + aux2 * gam1
        UJ[12] = aux * crss3 * dX3
    end

    return UJ
end

function cpu_vpm!(s, t)
    for i in 1:size(t, 2)
        for j in 1:size(s, 2)
            interaction!(t, s, i, j)
        end
    end
end

# Naive implementation
# Each thread handles a single target and uses global GPU memory
function gpu_vpm1!(s, t)
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

# Each thread handles a single target and uses local GPU memory
# Sources divided into multiple columns and influence is computed by multiple threads
function gpu_vpm3!(s, t, p, num_cols, kernel)
    t_size::Int32 = size(t, 2)
    s_size::Int32 = size(s, 2)

    ithread::Int32 = threadIdx().x

    # Row and column indices of threads in a block
    row = (ithread-1) % p + 1
    col = floor(Int32, (ithread-1)/p) + 1

    itarget::Int32 = row + (blockIdx().x-1)*p
    if itarget <= t_size
        @inbounds tx = t[1, itarget]
        @inbounds ty = t[2, itarget]
        @inbounds tz = t[3, itarget]
    end

    n_tiles::Int32 = CUDA.ceil(Int32, s_size / p)
    bodies_per_col::Int32 = CUDA.ceil(Int32, p / num_cols)

    sh_mem = CuDynamicSharedArray(eltype(t), (7, p))

    # Variable initialization
    UJ = @MVector zeros(eltype(t), 12)
    out = @MVector zeros(eltype(t), 12)
    idim::Int32 = 0
    idx::Int32 = 0
    i::Int32 = 0
    isource::Int32 = 0

    itile::Int32 = 1
    while itile <= n_tiles
        # Each thread will copy source coordinates corresponding to its index into shared memory. This will be done for each tile.
        if (col == 1)
            idx = row + (itile-1)*p
            idim = 1
            if idx <= s_size
                while idim <= 7
                    @inbounds sh_mem[idim, row] = s[idim, idx]
                    idim += 1
                end
            else
                while idim <= 7
                    @inbounds sh_mem[idim, row] = zero(eltype(s))
                    idim += 1
                end
            end
        end
        sync_threads()

        # Each thread will compute the influence of all the sources in the shared memory on the target corresponding to its index
        i = 1
        while i <= bodies_per_col
            isource = i + bodies_per_col*(col-1)
            if isource <= s_size
                if itarget <= t_size
                    out .= gpu_interaction(tx, ty, tz, sh_mem, isource, kernel)
                end

                # Sum up influences for each source in a tile
                idim = 1
                while idim <= 12
                    @inbounds UJ[idim] += out[idim]
                    idim += 1
                end
            end
            i += 1
        end
        itile += 1
        sync_threads()
    end

    # Sum up accelerations for each target/thread
    # Each target will be accessed by q no. of threads
    idx = 1
    if itarget <= t_size
        while idx <= 3
            @inbounds CUDA.@atomic t[9+idx, itarget] += UJ[idx]
            idx += 1
        end
        idx = 4
        while idx <= 12
            @inbounds CUDA.@atomic t[12+idx, itarget] += UJ[idx]
            idx += 1
        end
    end
    return
end

# High-storage parallel reduction
function gpu_vpm4!(s, t, num_cols, gb_mem, kernel)
    t_size::Int32 = size(t, 2)
    s_size::Int32 = size(s, 2)

    ithread::Int32 = threadIdx().x
    p::Int32 = t_size/gridDim().x

    # Row and column indices of threads in a block
    row = (ithread-1) % p + 1
    col = floor(Int32, (ithread-1)/p) + 1

    itarget::Int32 = row + (blockIdx().x-1)*p
    @inbounds tx = t[1, itarget]
    @inbounds ty = t[2, itarget]
    @inbounds tz = t[3, itarget]

    n_tiles::Int32 = CUDA.ceil(Int32, s_size / p)
    bodies_per_col::Int32 = CUDA.ceil(Int32, p / num_cols)

    sh_mem = CuDynamicSharedArray(eltype(t), (7, p))

    # Variable initialization
    UJ = @MVector zeros(eltype(t), 12)
    idim::Int32 = 0
    idx::Int32 = 0
    i::Int32 = 0
    isource::Int32 = 0

    itile::Int32 = 1
    while itile <= n_tiles
        # Each thread will copy source coordinates corresponding to its index into shared memory. This will be done for each tile.
        if (col == 1)
            idx = row + (itile-1)*p
            idim = 1
            if idx <= s_size
                while idim <= 7
                    @inbounds sh_mem[idim, row] = s[idim, idx]
                    idim += 1
                end
            else
                while idim <= 7
                    @inbounds sh_mem[idim, row] = zero(eltype(s))
                    idim += 1
                end
            end
        end
        sync_threads()

        # Each thread will compute the influence of all the sources in the shared memory on the target corresponding to its index
        i = 1
        while i <= bodies_per_col
            isource = i + bodies_per_col*(col-1)
            if isource <= s_size
                out = gpu_interaction(tx, ty, tz, sh_mem, isource, kernel)

                # Sum up influences for each source in a column in the tile
                # This UJ resides in the local memory of the thread corresponding
                # to each column, so we haven't summed up over the tile yet.
                idim = 1
                while idim <= 12
                    @inbounds UJ[idim] += out[idim]
                    idim += 1
                end
            end
            i += 1
        end
        itile += 1
        sync_threads()
    end

    # Sum up accelerations for each target/thread
    # Each target will be accessed by q no. of threads
    if num_cols != 1
        # Perform write to global memory
        # Columns correspond to each of the q threads
        # sh_mem[1:12, 1] is the first target, sh_mem[13:24, 1] is the second target and so on.
        idim = 1
        while idim <= 12
            @inbounds gb_mem[idim + 12*(itarget-1), col] = UJ[idim]
            idim += 1
        end
    else
        idim = 1
        while idim <= 3
            @inbounds t[9+idim, itarget] += UJ[idim]
            idim += 1
        end
        idim = 4
        while idim <= 12
            @inbounds t[12+idim, itarget] += UJ[idim]
            idim += 1
        end
    end

    return
end

function gpu_reduction!(gb_mem, t)
    tIdx::Int32 = threadIdx().x
    bDim::Int32 = blockDim().x
    bIdx::Int32 = blockIdx().x
    sh_mem = CuDynamicSharedArray(eltype(gb_mem), bDim)

    # Each thread copies content to shared memory
    @inbounds sh_mem[tIdx] = gb_mem[bIdx, tIdx]

    # Perform parallel reduction
    stride::Int32 = 1
    i::Int32 = 0
    while stride < bDim
        i = (tIdx-1)*stride*2+1
        if i <= bDim
            @inbounds sh_mem[i] += sh_mem[i+stride]
        end
        stride *= 2
        sync_threads()
    end

    # Copy from shared memory to target in global memory
    i = (bIdx-1) % 12 + 1
    stride = ceil(bIdx / 12)
    if tIdx == 1
        if i <= 3
            @inbounds t[9+i, stride] += sh_mem[1]
        else
            @inbounds t[12+i, stride] += sh_mem[1]
        end
    end
    return
end

# Each thread handles a single target and uses local GPU memory
# Sources divided into multiple columns and influence is computed by multiple threads
# Final summation through parallel reduction instead of atomic reduction
# Low-storage parallel reduction
# - p is no. of targets per block. Typically same as no. of sources per block.
# - q is no. of columns per tile
function gpu_vpm5!(s, t, num_cols, kernel)
    t_size::Int32 = size(t, 2)
    s_size::Int32 = size(s, 2)

    ithread::Int32 = threadIdx().x
    p::Int32 = t_size/gridDim().x

    # Row and column indices of threads in a block
    row = (ithread-1) % p + 1
    col = floor(Int32, (ithread-1)/p) + 1

    itarget::Int32 = row + (blockIdx().x-1)*p
    @inbounds tx = t[1, itarget]
    @inbounds ty = t[2, itarget]
    @inbounds tz = t[3, itarget]

    n_tiles::Int32 = CUDA.ceil(Int32, s_size / p)
    bodies_per_col::Int32 = CUDA.ceil(Int32, p / num_cols)

    sh_mem = CuDynamicSharedArray(eltype(t), (12, p))

    # Variable initialization
    UJ = @MVector zeros(eltype(t), 12)
    out = @MVector zeros(eltype(t), 12)
    idim::Int32 = 0
    idx::Int32 = 0
    i::Int32 = 0
    isource::Int32 = 0

    itile::Int32 = 1
    while itile <= n_tiles
        # Each thread will copy source coordinates corresponding to its index into shared memory. This will be done for each tile.
        if (col == 1)
            idx = row + (itile-1)*p
            idim = 1
            if idx <= s_size
                while idim <= 7
                    @inbounds sh_mem[idim, row] = s[idim, idx]
                    idim += 1
                end
            else
                while idim <= 7
                    @inbounds sh_mem[idim, row] = zero(eltype(s))
                    idim += 1
                end
            end
        end
        sync_threads()

        # Each thread will compute the influence of all the sources in the shared memory on the target corresponding to its index
        i = 1
        while i <= bodies_per_col
            isource = i + bodies_per_col*(col-1)
            if isource <= s_size
                out .= gpu_interaction(tx, ty, tz, sh_mem, isource, kernel)

                # Sum up influences for each source in a column in the tile
                # This UJ resides in the local memory of the thread corresponding
                # to each column, so we haven't summed up over the tile yet.
                idim = 1
                while idim <= 12
                    @inbounds UJ[idim] += out[idim]
                    idim += 1
                end
            end
            i += 1
        end
        itile += 1
        sync_threads()
    end

    # Sum up accelerations for each target/thread
    # Each target will be accessed by q no. of threads
    if num_cols != 1
        # Perform write to shared memory
        # Columns correspond to each of the q threads
        # Iterate over targets and do reduction
        it::Int32 = 1
        while it <= p
            # Threads corresponding to itarget will copy their data to shared mem
            if itarget == it+p*(blockIdx().x-1)
                idim = 1
                while idim <= 12
                    @inbounds sh_mem[idim, col] = UJ[idim]
                    idim += 1
                end
            end
            sync_threads()

            # All p*q threads do parallel reduction on data
            stride::Int32 = 1
            while stride < num_cols
                i = (threadIdx().x-1)*stride*2+1
                if i+stride <= num_cols
                    idim = 1
                    while idim <= 12  # This can be parallelized too
                        @inbounds sh_mem[idim, i] += sh_mem[idim, i+stride]
                        idim += 1
                    end
                end
                stride *= 2
                sync_threads()
            end

            # col 1 of the threads that handle 'it' target
            # writes reduced data to its own local memory
            if itarget == it+p*(blockIdx().x-1) && col == 1
                idim = 1
                while idim <= 12
                    @inbounds UJ[idim] = sh_mem[idim, 1]
                    idim += 1
                end
            end

            it += 1
        end
    end

    # Now, each col 1 has the net influence of all sources on its target
    # Write all data back to global memory
    if col == 1
        idim = 1
        while idim <= 3
            @inbounds t[9+idim, itarget] += UJ[idim]
            idim += 1
        end
        idim = 4
        while idim<= 12
            @inbounds t[12+idim, itarget] += UJ[idim]
            idim += 1
        end
    end

    return
end

# Each thread handles a single target and uses local GPU memory
# Sources divided into multiple columns and influence is computed by multiple threads
# More sources into shared memory
function gpu_vpm6!(s, t, p, num_cols, kernel)
    t_size::Int32 = size(t, 2)
    s_size::Int32 = size(s, 2)

    ithread::Int32 = threadIdx().x

    # Row and column indices of threads in a block
    row = (ithread-1) % p + 1
    col = floor(Int32, (ithread-1)/p) + 1

    itarget::Int32 = row + (blockIdx().x-1)*p
    if itarget <= t_size
        @inbounds tx = t[1, itarget]
        @inbounds ty = t[2, itarget]
        @inbounds tz = t[3, itarget]
    end

    n_tiles::Int32 = CUDA.ceil(Int32, s_size / (p*num_cols))
    bodies_per_col::Int32 = p

    sh_mem = CuDynamicSharedArray(eltype(t), (7, p*num_cols))

    # Variable initialization
    UJ = @MVector zeros(eltype(t), 12)
    out = @MVector zeros(eltype(t), 12)
    idim::Int32 = 0
    idx::Int32 = 0
    i::Int32 = 0
    isource::Int32 = 0

    itile::Int32 = 1
    while itile <= n_tiles
        @cushow itile
        # Each thread will copy source coordinates corresponding to its index into shared memory. This will be done for each tile.
        idx = ithread + (itile-1)*p*num_cols
        idim = 1
        if idx <= s_size
            while idim <= 7
                @inbounds sh_mem[idim, idx] = s[idim, idx]
                idim += 1
            end
        else
            while idim <= 7
                @inbounds sh_mem[idim, row] = zero(eltype(s))
                idim += 1
            end
        end
        sync_threads()

        # Each thread will compute the influence of all the sources in the shared memory on the target corresponding to its index
        i = 1
        while i <= bodies_per_col
            isource = i + bodies_per_col*(col-1)
            if isource <= s_size
                if itarget <= t_size
                    out .= gpu_interaction(tx, ty, tz, sh_mem, isource, kernel)
                end

                # Sum up influences for each source in a tile
                idim = 1
                while idim <= 12
                    @inbounds UJ[idim] += out[idim]
                    idim += 1
                end
            end
            i += 1
        end
        itile += 1
        sync_threads()
    end

    # Sum up accelerations for each target/thread
    # Each target will be accessed by q no. of threads
    idx = 1
    if itarget <= t_size
        while idx <= 3
            @inbounds CUDA.@atomic t[9+idx, itarget] += UJ[idx]
            idx += 1
        end
        idx = 4
        while idx <= 12
            @inbounds CUDA.@atomic t[12+idx, itarget] += UJ[idx]
            idx += 1
        end
    end
    return
end


function benchmark1_gpu!(s, t)
    s_d = CuArray(view(s, 1:7, :))
    t_d = CuArray(t)

    kernel = @cuda launch=false gpu_vpm1!(s_d, t_d)
    config = launch_configuration(kernel.fun)
    threads = min(size(t, 2), config.threads)
    blocks = cld(size(t, 2), threads)

    kernel(s_d, t_d; threads, blocks)

    view(t, 10:12, :) .= Array(t_d[10:12, :])
    view(t, 16:24, :) .= Array(t_d[16:24, :])
end

function benchmark3_gpu!(s, t, p, q; t_padding=0)
    s_d = CuArray(view(s, 1:7, :))
    t_d = CuArray(view(t, 1:24, :))

    t_size = size(t_d, 2)+t_padding
    kernel = gpu_g_dgdr

    # Num of threads in a tile should always be 
    # less than number of threads in a block (1024)
    # or limited by memory size
    threads::Int32 = p*q
    blocks::Int32 = cld(t_size, p)
    shmem = sizeof(eltype(s)) * 7 * p  # XYZ + Γ123 + σ = 7 variables
    @cuda threads=threads blocks=blocks shmem=shmem gpu_vpm3!(s_d, t_d, p, q, kernel)

    view(t, 10:12, :) .= Array(view(t_d, 10:12, :))
    view(t, 16:24, :) .= Array(view(t_d, 16:24, :))
end

function benchmark4_gpu!(s, t, p, q)
    s_d = CuArray(view(s, 1:7, :))
    t_d = CuArray(view(t, 1:24, :))
    t_size = size(t_d, 2)
    kernel = gpu_g_dgdr

    # Num of threads in a tile should always be 
    # less than number of threads in a block (1024)
    # or limited by memory size
    threads::Int32 = p*q
    blocks::Int32 = cld(t_size, p)
    shmem = sizeof(eltype(s)) * 7 * p  # XYZ + Γ123 + σ = 7 variables but (12*p) to handle UJ summation for each target
    gb_mem = CUDA.zeros(eltype(t_d), 12*t_size, q)
    @cuda threads=threads blocks=blocks shmem=shmem gpu_vpm4!(s_d, t_d, q, gb_mem, kernel)

    # Parallel reduction on 12p targets, with q partial influences
    shmem = sizeof(eltype(s)) * p
    threads = q
    blocks = 12*t_size
    @cuda threads=threads blocks=blocks shmem=shmem gpu_reduction!(gb_mem, t_d)

    view(t, 10:12, :) .= Array(t_d[10:12, :])
    view(t, 16:24, :) .= Array(t_d[16:24, :])
end

function benchmark5_gpu!(s, t, p, q)
    s_d = CuArray(view(s, 1:7, :))
    t_d = CuArray(view(t, 1:24, :))
    kernel = gpu_g_dgdr

    # Num of threads in a tile should always be 
    # less than number of threads in a block (1024)
    # or limited by memory size
    threads::Int32 = p*q
    blocks::Int32 = cld(size(t, 2), p)
    shmem = sizeof(eltype(s)) * 12 * p  # XYZ + Γ123 + σ = 7 variables but (12*p) to handle UJ summation for each target
    @cuda threads=threads blocks=blocks shmem=shmem gpu_vpm5!(s_d, t_d, q, kernel)

    view(t, 10:12, :) .= Array(t_d[10:12, :])
    view(t, 16:24, :) .= Array(t_d[16:24, :])
end

function benchmark6_gpu!(s, t, p, q; t_padding=0)
    s_d = CuArray(view(s, 1:7, :))
    t_d = CuArray(view(t, 1:24, :))

    t_size = size(t_d, 2)+t_padding
    kernel = gpu_g_dgdr

    # Num of threads in a tile should always be 
    # less than number of threads in a block (1024)
    # or limited by memory size
    threads::Int32 = p*q
    blocks::Int32 = cld(t_size, p)
    shmem = sizeof(eltype(s)) * 7 * p*q  # XYZ + Γ123 + σ = 7 variables
    @cuda threads=threads blocks=blocks shmem=shmem gpu_vpm6!(s_d, t_d, p, q, kernel)

    view(t, 10:12, :) .= Array(view(t_d, 10:12, :))
    view(t, 16:24, :) .= Array(view(t_d, 16:24, :))
end


function check_launch(n, p, q; throw_error=true, max_threads_per_block=512)
    isgood = true

    if p > n; isgood = false; throw_error && error("p must be less than or equal to n"); end
    if p*q >= max_threads_per_block; isgood = false; throw_error && error("p*q must be less than $max_threads_per_block"); end
    if q > p; isgood = false; throw_error && error("q must be less than or equal to p"); end
    if n % p != 0; isgood = false; throw_error && error("n must be divisible by p"); end
    if p % q != 0; isgood = false; throw_error && error("p must be divisible by q"); end

    return isgood
end

function main(run_option; ns=2^5, nt=0, p=0, q=1, debug=false, padding=true, max_threads_per_block=512)
    T = Float64

    nt = nt==0 ? ns : nt

    t_padding = 0
    # Pad target array to nearest multiple of 10 for efficient p, q launch configuration
    if padding && mod(nt, 10) > 0
        t_padding =  10 - mod(nt, 10)
    end
    if p == 0
        p, q = get_launch_config(nt+t_padding; max_threads_per_block=max_threads_per_block)
        # @show p, q
    end
    if run_option == 1 || run_option == 2
        println("No. of sources: $ns")
        println("No. of targets: $nt")
        println("Tile length, p: $p")
        println("Cols per tile, q: $q")

        check_launch(nt+t_padding, p, q; max_threads_per_block=max_threads_per_block)

        src, trg, src2, trg2 = get_inputs(ns, nfields; T=T, nt=nt)
        if run_option == 1
            println("CPU Run")
            cpu_vpm!(src, trg)
            println("GPU Run")
            # benchmark3_gpu!(src2, trg2, p, q; t_padding=t_padding)
            # benchmark4_gpu!(src2, trg2, p, q)
            # benchmark5_gpu!(src2, trg2, p, q)
            benchmark6_gpu!(src2, trg2, p, q)
            diff = abs.(trg .- trg2)
            err_norm = sqrt(sum(abs2, diff)/length(diff))
            diff_bool = diff .< eps(T)
            if all(diff_bool)
                println("MATCHES")
            else
                # Write to file to check errors
                # writedlm("trg_cpu.dat", trg[10, :])
                # writedlm("trg_gpu.dat", trg2[10, :])
                if ns < 10 && debug
                    display(trg[10:12, :])
                    display(trg2[10:12, :])
                    display(diff[10:12, :])
                    # println("J vals")
                    # display(trg[16:24, :])
                    # display(trg2[16:24, :])
                    # display(diff[16:24, :])
                end
                n_diff = count(==(false), diff_bool)
                n_total = size(trg, 1)*size(trg, 2)
                println("$n_diff of $n_total elements DO NOT MATCH")
                println("Error norm: $err_norm")
            end
        else
            println("Running profiler...")
            CUDA.@profile external=true benchmark3_gpu!(src2, trg2, p, q; t_padding=t_padding)
            # CUDA.@profile external=true benchmark4_gpu!(src2, trg2, p, q)
            # CUDA.@profile external=true benchmark5_gpu!(src2, trg2, p, q)
        end
    else
        check_launch(nt+t_padding, p, q, max_threads_per_block=max_threads_per_block)

        src, trg, src2, trg2 = get_inputs(ns, nfields)
        t_cpu = @benchmark cpu_vpm!($src, $trg)
        t_gpu = @benchmark benchmark3_gpu!($src2, $trg2, $p, $q; t_padding=$t_padding)
        # t_gpu = @benchmark benchmark4_gpu!($src2, $trg2, $p, $q)
        # t_gpu = @benchmark benchmark5_gpu!($src2, $trg2, $p, $q)
        speedup = median(t_cpu.times)/median(t_gpu.times)
        println("$ns $speedup")
    end
    return
end

function get_launch_config(nt; p_max=512, max_threads_per_block=512)
    divs_n = sort(divisors(nt))
    p = 1
    q = 1
    ip = 1
    for (i, div) in enumerate(divs_n)
        if div <= p_max
            p = div
            ip = i
        end
    end

    # Decision algorithm 1: Creates a matrix using indices and finds max of 
    # weighted sum of indices

    i_weight = 0
    j_weight = 1-i_weight

    max_ij = i_weight*ip + j_weight*1
    if nt <= 2^13
        divs_p = divs_n
        for i in 1:length(divs_n)
            for j in 1:length(divs_n)
                isgood = check_launch(nt, divs_n[i], divs_p[j]; throw_error=false, max_threads_per_block=max_threads_per_block)
                if isgood && (divs_n[i] <= p_max)
                    # Check if this is the max achievable ij value
                    # in the p, q choice matrix
                    obj_val = i_weight*i+j_weight*j
                    if obj_val >= max_ij
                        max_ij = obj_val
                        p = divs_n[i]
                        q = divs_p[j]
                    end
                end
            end
        end
    end

    return p, q
end

# Run_option - # [1]test [2]profile [3]benchmark
# for i in 7:17
#     main(3; ns=2^i, T=Float32)
# end
# main(1; ns=2, debug=true)
# main(3; ns=2^9, nt=2^12, T=Float32, debug=true)
# main(1; ns=8739, nt=3884, debug=true)
# main(1; ns=33, p=11, T=Float64)
# main(1; ns=130, p=26, q=2, T=Float64)
# main(1; ns=8, p=4, q=2, padding=false, debug=true)
# main(3, ns=1459; debug=true)
# main(3, ns=1460; debug=true)
# main(3, ns=1579; debug=true)
# main(3, ns=1480; debug=true)
main(1, ns=3, nt=2, p=2, q=1, padding=false)
