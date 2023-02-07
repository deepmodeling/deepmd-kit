#!/bin/bash
#SBATCH -J srs-lmp
#SBATCH --gpus 1
 
# export PATH=/data/run01/scv6266/soft/lammps-DP/src:$PATH
export PATH=/data/run01/scv6266/soft/lammps_23Jun2022/src:$PATH
mpirun -np 1 lmp_mpi -in in.force
