#!/bin/bash
#SBATCH --job-name=Assg1_Parall5l
#SBATCH -N 2
#SBATCH --ntasks-per-node=16
#SBATCH --output=./SlurmOutputs/assg1_%j.out
#SBATCH --error=./Errors/assg1_%j.err
#SBATCH --partition=cpu
#SBATCH --time=00:05:00

# Load MPI module
module load compiler/oneapi-2024/mpi

# Parameters
P_LIST="8 16 32"
M_LIST="262144 1048576"

# Repeat each configuration 5 times
for run in {1..5}
do
    echo "Run $run"

    for P in $P_LIST
    do
        for M in $M_LIST
        do
                for D1 in 2
                do
                        for D2 in 4
                        do
                                for T in 10
                                do
                                        for SEED in 1000
                                        do
                                                echo "P=$P M=$M D1=$D1 D2=$D2 T=$T SEED=$SEED"

                                                mpirun -np $P ./src.exe $M $D1 $D2 $T $SEED | tee -a ${SLURM_JOBID}_out.txt
                                        done
                                done
                        done
                done
        done
    done
done

mv ${SLURM_JOBID}_out.txt ./Outputs/