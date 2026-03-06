#!/usr/bin/env bash

# ======== Slurm setting
#SBATCH -J pga_ft_s12mae_zf
#SBATCH --output=pga_ft_s12mae_zf_%j.out
#SBATCH --error=pga_ft_s12mae_zf_%j.err
#SBATCH -D /dss/dsshome1/07/di54rur/kim_he/ZeroFlood
#SBATCH --time=12:00:00

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
micromamba activate pangaea-bench
export PYTHONNOUSERSITE=1
export MASTER_PORT=$((12000 + SLURM_JOB_ID % 20000))
export HYDRA_FULL_ERROR=1

# Set root path
SOURCE_ROOT=/dss/dsshome1/07/di54rur/kim_he/pangaea-bench
OUTPUT_ROOT=/dss/dsstbyfs02/scratch/07/di54rur/pangaea-bench/
MODEL_ROOT=/dss/dsstbyfs02/scratch/07/di54rur/pangaea-bench/pretrained_models/B2_vits16_mae_ep99.pth
DATASET_ROOT=/dss/dsstbyfs02/scratch/07/di54rur/zeroflood/ZeroFlood

cd $SOURCE_ROOT

# Launch training
srun torchrun --nnodes=1 --nproc_per_node=1 --master_port $MASTER_PORT \
   pangaea/run.py \
   --config-name=train \
   work_dir=$OUTPUT_ROOT \
   dataset=zeroflood_sar \
   dataset.root_path=$DATASET_ROOT \
   encoder=ssl4eo_mae_sar \
   encoder.encoder_weights=$MODEL_ROOT \
   decoder=seg_upernet \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation \
   task.trainer.log_interval=100 \
   batch_size=16