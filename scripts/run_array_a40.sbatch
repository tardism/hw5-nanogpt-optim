#!/bin/bash
#SBATCH --job-name=nano-gpt-opt
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --array=0-8%2
#SBATCH --output=logs/%x_%A_%a.out

# === Adjust these if your MSI environment uses different modules ===
module purge
module load cuda/12.1

# Activate your python env (edit path)
source ~/nanogpt-env/bin/activate

# Map TASK_ID -> (optimizer, lr)
OPTIMIZERS=(adam rmsprop signsgd)
# You can edit LR candidates per optimizer here:
LRS_ADAM=(1e-4 3e-4 1e-3)
LRS_RMSPROP=(1e-4 1e-3 3e-3)
LRS_SIGNSGD=(1e-3 1e-2 3e-2)

TASK_ID=${SLURM_ARRAY_TASK_ID}
OPT_ID=$((TASK_ID / 3))
LR_ID=$((TASK_ID % 3))
OPT=${OPTIMIZERS[$OPT_ID]}

if [ "$OPT" = "adam" ]; then
  LR=${LRS_ADAM[$LR_ID]}
elif [ "$OPT" = "rmsprop" ]; then
  LR=${LRS_RMSPROP[$LR_ID]}
else
  LR=${LRS_SIGNSGD[$LR_ID]}
fi

OUT_DIR=runs/${OPT}_lr${LR}
mkdir -p "$OUT_DIR"

echo "Running: optimizer=$OPT lr=$LR out_dir=$OUT_DIR"
python src/train.py \
  --config configs/base.yaml \
  --data_dir data \
  --out_dir "$OUT_DIR" \
  --optimizer "$OPT" \
  --lr "$LR" \
  --steps 5000 \
  --eval_interval 200 \
  --eval_iters 50 \
  --device cuda
