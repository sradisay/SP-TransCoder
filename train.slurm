#!/bin/bash
#SBATCH --job-name=translate
#SBATCH -A cs175a_class_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --output=logs/translate.log
#SBATCH --error=logs/translate.err

eval "$(/opt/apps/miniconda3/24.9.2/bin/conda shell.bash hook)"
conda activate tdmpc2

nvidia-smi

export PYTHONUNBUFFERED=1
python main.py
