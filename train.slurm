#!/bin/bash
#SBATCH --job-name=train_cifar
#SBATCH --partition=free-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=48:00:00
#SBATCH --output=logs/translate.log
#SBATCH --error=logs/translate.err

eval "$(/opt/apps/miniconda3/24.9.2/bin/conda shell.bash hook)"
conda activate tdmpc2

python main.py
