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
PRETRAINED=checkpoints/pretrained-5x5.keras
FIRST=results/phase4/learn-5x5/checkpoints/iteration-0000.keras
conda run -n zur_env python -m breakthrough_zero.cli arena \
  --agent-a neural --a-checkpoint "$LATEST" --a-name latest \
  --agent-b random --b-name random \
  --board-size 5 --starting-rows 1 \
  --openings 50 --prefix-plies 4 --move-seconds 0.1 \
  --output results/phase5/neural-v-random.json
conda run -n zur_env python -m breakthrough_zero.cli arena \
  --agent-a neural --a-checkpoint "$LATEST" --a-name latest \
  --agent-b alphabeta --b-name alpha-beta \
  --board-size 5 --starting-rows 1 \
  --openings 50 --prefix-plies 4 --move-seconds 0.1 \
  --output results/phase5/neural-v-alphabeta.json
conda run -n zur_env python -m breakthrough_zero.cli arena \
  --agent-a neural --a-checkpoint "$LATEST" --a-name latest \
  --agent-b rollout --b-name rollout-puct \
  --board-size 5 --starting-rows 1 \
  --openings 50 --prefix-plies 4 --move-seconds 0.1 \
  --output results/phase5/neural-v-rollout.json
conda run -n zur_env python -m breakthrough_zero.cli arena \
  --agent-a neural --a-checkpoint "$PRETRAINED" --a-name pretrained \
  --agent-b rollout --b-name rollout-puct \
  --board-size 5 --starting-rows 1 \
  --openings 50 --prefix-plies 4 --move-seconds 0.1 \
  --output results/phase5/pretrained-v-rollout.json
conda run -n zur_env python -m breakthrough_zero.cli arena \
  --agent-a neural --a-checkpoint "$LATEST" --a-name latest \
  --agent-b neural --b-checkpoint "$PRETRAINED" --b-name pretrained \
  --board-size 5 --starting-rows 1 \
  --openings 50 --prefix-plies 4 --move-seconds 0.1 \
  --output results/phase5/latest-v-pretrained.json
conda run -n zur_env python -m breakthrough_zero.cli arena \
  --agent-a neural --a-checkpoint "$LATEST" --a-name latest \
  --agent-b neural --b-checkpoint "$FIRST" --b-name iteration-0000 \
  --board-size 5 --starting-rows 1 \
  --openings 50 --prefix-plies 4 --move-seconds 0.1 \
  --output results/phase5/latest-v-iteration0000.json
conda run -n zur_env python -m breakthrough_zero.cli elo-table \
  --reports results/phase5/*-v-*.json \
  --anchor random \
  --output results/phase5/elo-table.json
