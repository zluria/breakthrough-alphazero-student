#!/bin/bash
#SBATCH --job-name=bt-best-v-ab
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/%x-%j.out

set -euo pipefail
module load anaconda
PROJECT_DIR="$(pwd)"
export PYTHONPATH="$PROJECT_DIR/src"

# Iteration 26 is the latest checkpoint that passed a scheduled strength match.
# Twenty-five openings are each played with both color assignments, giving 50
# games while separating playing strength from first-player and opening luck.
CHECKPOINT=results/phase8/serious-az-8x8/checkpoints/iteration-0026.keras
OUTPUT=results/phase8/serious-az-8x8/best-v-alphabeta-50.json

test -f "$CHECKPOINT"
mkdir -p results/slurm results/phase8/serious-az-8x8
conda run -n zur_env python -m breakthrough_zero.cli arena \
  --agent-a neural --a-checkpoint "$CHECKPOINT" --a-name serious-iteration-26 \
  --agent-b alphabeta --b-name alpha-beta \
  --board-size 8 --starting-rows 2 \
  --openings 25 --prefix-plies 6 --move-seconds 0.1 \
  --output "$OUTPUT"
