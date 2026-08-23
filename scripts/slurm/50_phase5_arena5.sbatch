#!/bin/bash
#SBATCH --job-name=bt-arena5
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm/%x-%j.out

set -euo pipefail
module load anaconda
PROJECT_DIR="$(pwd)"
export PYTHONPATH="$PROJECT_DIR/src"

mkdir -p results/slurm results/phase5
LATEST=results/phase4/learn-5x5/checkpoints/latest.keras
conda run -n zur_env python -m breakthrough_zero.cli arena \
  --agent-a neural --a-checkpoint "$LATEST" \
  --agent-b random \
  --board-size 5 --starting-rows 1 \
  --openings 50 --prefix-plies 4 --move-seconds 0.1 \
  --output results/phase5/neural-v-random.json
conda run -n zur_env python -m breakthrough_zero.cli arena \
  --agent-a neural --a-checkpoint "$LATEST" \
  --agent-b alphabeta \
  --board-size 5 --starting-rows 1 \
  --openings 50 --prefix-plies 4 --move-seconds 0.1 \
  --output results/phase5/neural-v-alphabeta.json
conda run -n zur_env python -m breakthrough_zero.cli arena \
  --agent-a neural --a-checkpoint "$LATEST" \
  --agent-b rollout \
  --board-size 5 --starting-rows 1 \
  --openings 50 --prefix-plies 4 --move-seconds 0.1 \
  --output results/phase5/neural-v-rollout.json
