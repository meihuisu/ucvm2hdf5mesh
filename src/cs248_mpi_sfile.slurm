#!/bin/bash
#SBATCH -A geo156 
#SBATCH -J cs248_mpi_sfile
#SBATCH -o cs248_mpi_sfile.oo
#SBATCH -e cs248_mpi_sfile.ee
#SBATCH -p batch
#SBATCH -t 2:00:00
#SBATCH -N 140 
#SBATCH --mail-user=mei@usc.edu
#SBATCH --mail-type=all
#SBATCH --open-mode=truncate

cd $SLURM_SUBMIT_DIR
echo "slurm start"
date

srun -n2801 cs248_mpi_sfile.py 
echo "slurm done"
date
