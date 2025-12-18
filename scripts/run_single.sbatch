#!/bin/bash
#SBATCH --job-name=nano-gpt-debug
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%A.out

module purge
module load cuda/12.1
source ~/nanogpt-env/bin/activate

python3 src/train.py --steps 200 --eval_interval 50 --out_dir runs/debug --optimizer adam --lr 3e-4 --device cuda
