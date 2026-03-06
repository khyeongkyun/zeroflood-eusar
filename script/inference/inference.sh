#!/usr/bin/env bash

# ======== Slurm setting
#SBATCH -J inf_*
#SBATCH --output=inf_*_%j.out
#SBATCH --error=inf_*_%j.err
#SBATCH -D /dss/dsshome1/07/di54rur/kim_he/zeroflood-eusar
#SBATCH --time=1:00:00

# ======== General resourece setting
#SBATCH --mem=50gb
#SBATCH --clusters=hpda2
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --cpus-per-task=4

# ======== GPU resource setting
#SBATCH --gres=gpu:1

# load required modules
module load slurm_setup
eval "$(micromamba shell hook --shell bash)"
micromamba activate zeroflood-eusar
export PYTHONNOUSERSITE=1
export MASTER_PORT=$((12000 + SLURM_JOB_ID % 20000))
export HYDRA_FULL_ERROR=1

# Set root path
SOURCE_ROOT=/dss/dsshome1/07/di54rur/kim_he/zeroflood-eusar
CKPT_DIR=/dss/dsstbyfs02/scratch/07/di54rur/zeroflood/models/zeroflood-baseline-vit

cd $SOURCE_ROOT

# Launch training
srun torchrun --nnodes=1 --nproc_per_node=1 --master_port $MASTER_PORT \
   pangaea/run.py \
   --config-name=test \
   ckpt_dir=$CKPT_DIR \
   save_pred=true