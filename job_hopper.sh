#!/bin/bash

#SBATCH --time=70:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=2000M   # memory per CPU core
#SBATCH --gpus=1
#SBATCH -J "b-pq-hopp"   # job name
#SBATCH --qos=standby
#SBATCH --partition=cs2


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# julia --project=. optimalpq.jl | tee max.log
# julia --project=. benchmark_pq32.jl 5000 | tee out5000_32_try2.csv
# julia --project=. benchmark_pq.jl 5000 | tee out5000_try2.csv
# julia --project=. benchmark_pqr.jl | tee out32x5000.csv
# nt/ns < 1
# julia --project=. benchmark_pqr.jl 128 5000 | tee out128x5000.csv
# julia --project=. benchmark_pqr.jl 128 17100 | tee out128x17100.csv
# nt/ns > 1
# julia --project=. benchmark_pqr.jl 5000 128 | tee out5000x128.csv
# julia --project=. benchmark_pqr.jl 17100 128 | tee out17100x128.csv
# Higher no. of targets, nt/ns > 1
# julia --project=. benchmark_pqr.jl 1024 17100 | tee out1024x17100.csv
# julia --project=. benchmark_pq.jl 40000 | tee out40000_hop.csv


julia --project=. benchmark_pqr2.jl | tee pqr512_p2_hop.csv
julia --project=. benchmark_pqr2.jl 1024 | tee pqr1024_p2_hop.csv
