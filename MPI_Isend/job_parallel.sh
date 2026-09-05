#!/bin/bash -l
#SBATCH --qos=math-454
#SBATCH --account=math-454
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --time=00:30:00
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module purge 
module load gcc hdf5 openmpi
srun ./swe 1476 1477
