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

mkdir -p logs
mkdir -p checkpoints

export PYTHONUNBUFFERED=1
DROPOUT_PS=(0.5 0.75 1.0)
DROPOUT_P_IDX=${SLURM_ARRAY_TASK_ID}
DROPOUT_P=${DROPOUT_PS[DROPOUT_P_IDX]}

python ticket.py --dropout_p $DROPOUT_P
