#!/usr/bin/env bash

# ======== Slurm setting
#SBATCH -J pga_ft_tr_b_zf_tim_chain_sd
#SBATCH --output=pga_ft_tr_b_zf_tim_chain_sd_%j.out
#SBATCH --error=pga_ft_tr_b_zf_tim_chain_sd_%j.err
#SBATCH -D /dss/dsshome1/07/di54rur/kim_he/zeroflood-eusar
#SBATCH --time=2-12:00:00

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
OUTPUT_ROOT=/dss/dsstbyfs02/scratch/07/di54rur/pangaea-bench/
MODEL_ROOT=/dss/dsstbyfs02/scratch/07/di54rur/pangaea-bench/pretrained_models/TerraMind_v1_base.pt
DATASET_ROOT=/dss/dsstbyfs02/scratch/07/di54rur/zeroflood/ZeroFlood

cd $SOURCE_ROOT

# Launch training
srun torchrun --nnodes=1 --nproc_per_node=1 --master_port $MASTER_PORT \
   pangaea/run.py \
   --config-name=train \
   work_dir=$OUTPUT_ROOT \
   dataset=zeroflood_optical \
   dataset.root_path=$DATASET_ROOT \
   encoder=terramind_base_optical_tim \
   encoder.tim_chained_generations=True \
   encoder.encoder_weights=$MODEL_ROOT \
   'encoder.tim_modalities=[S1RTC,DEM]' \
   decoder=seg_upernet \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation \
   task.trainer.log_interval=100 \
   batch_size=16
