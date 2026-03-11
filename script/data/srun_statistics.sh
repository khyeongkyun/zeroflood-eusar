#!/usr/bin/env bash

# ======== Slurm setting
#SBATCH -J statistics
#SBATCH --output=statistics_%j.out
#SBATCH --error=statistics_%j.err
#SBATCH -D /dss/dsshome1/07/di54rur/kim_he/zeroflood-eusar
#SBATCH --time=5:00:00

# ======== General resourece setting
#SBATCH --mem=64gb
#SBATCH --clusters=hpda2
#SBATCH --partition=hpda2_compute
#SBATCH --cpus-per-task=4

# load required modules
module load slurm_setup
eval "$(conda shell.bash hook)"
conda activate zeroflood-eusar

# run script as slurm job
export PYTHONPATH=$PWD/src:$PYTHONPATH
srun python ./data/statistics.py