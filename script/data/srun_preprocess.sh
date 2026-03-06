#!/usr/bin/env bash

# ======== Slurm setting
#SBATCH -J preprocess
#SBATCH --output=preprocess_%j.out
#SBATCH --error=preprocess_%j.err
#SBATCH -D /dss/dsshome1/07/di54rur/kim_he/ZeroFlood
#SBATCH --time=1-12:00:00

# ======== General resourece setting
#SBATCH --mem=64gb
#SBATCH --clusters=hpda2
#SBATCH --partition=hpda2_compute
#SBATCH --cpus-per-task=4

# load required modules
module load slurm_setup
eval "$(conda shell.bash hook)"
conda activate zeroflood

# run script as slurm job
export PYTHONPATH=$PWD/src:$PYTHONPATH
srun python ./data/preprocess.py -s all -r /dss/dsstbyfs02/scratch/07/di54rur/zeroflood