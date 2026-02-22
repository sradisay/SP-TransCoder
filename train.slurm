#!/bin/bash
#SBATCH --job-name=train_cifar
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --output=logs/translate.log
#SBATCH --error=logs/translate.err

# eval "$(/opt/apps/miniconda3/24.9.2/bin/conda shell.bash hook)"
eval "$(/pkg/anaconda3/2025.06/bin/conda shell.bash hook)"
conda activate tdmpc2

export HF_HOME=/scratch/$USER/hf_home
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache
mkdir -p $HF_HOME $HF_DATASETS_CACHE

nvidia-smi

export PYTHONUNBUFFERED=1
python main.py
