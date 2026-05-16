using CUDA, MPI

function kernel!(a, idev)
    a[idev] = idev
    return
end

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    CUDA.device!(rank % length(CUDA.devices()))

    a_d = CUDA.zeros(nranks)
    @cuda threads=1 kernel!(a_d, rank + 1)
    CUDA.synchronize()

    a = Array(a_d)
    # Reduce across ranks if you want the full array on all ranks
    MPI.Allreduce!(a, +, comm)

    if rank == 0
        println(a)
    end

    MPI.Finalize()
end

main()
