#!/bin/bash
#SBATCH --job-name=SP-TransCoder_semi
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH --time=48:00:00
#SBATCH --output=logs/semi_supervised_with_denoising.log
#SBATCH --error=logs/semi_supervised_with_denoising.err

eval "$(/opt/apps/miniconda3/24.9.2/bin/conda shell.bash hook)"
conda activate tdmpc2

python train_semi_supervised_with_denoising.py