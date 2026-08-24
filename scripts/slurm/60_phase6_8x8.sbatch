#!/bin/bash
#SBATCH --job-name=bt-8x8
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=results/slurm/%x-%j.out

set -euo pipefail
module load anaconda
PROJECT_DIR="$(pwd)"
export PYTHONPATH="$PROJECT_DIR/src"

mkdir -p results/slurm results/phase6 data/raw checkpoints
conda run -n zur_env python -m breakthrough_zero.cli pretrain-data \
  --config configs/pretrain-8x8.json \
  --output data/raw/pretraining-8x8-10000.jsonl.gz \
  --report results/phase6/pretraining-data-report.json
conda run -n zur_env python -m breakthrough_zero.cli pretrain-network \
  --data data/raw/pretraining-8x8-10000.jsonl.gz \
  --output checkpoints/pretrained-8x8.keras \
  --report results/phase6/pretraining-network-report.json \
  --epochs 8 --batch-size 128
conda run -n zur_env python -m breakthrough_zero.cli learn \
  --config configs/learn-8x8.json \
  --run-dir results/phase6/learn-8x8 \
  --checkpoint checkpoints/pretrained-8x8.keras \
  --report results/phase6/learn-8x8-report.json
LATEST8=results/phase6/learn-8x8/checkpoints/latest.keras
conda run -n zur_env python -m breakthrough_zero.cli arena \
  --agent-a neural --a-checkpoint "$LATEST8" --a-name latest-8x8 \
  --agent-b random --b-name random \
  --board-size 8 --starting-rows 2 \
  --openings 30 --prefix-plies 6 --move-seconds 0.1 \
  --output results/phase6/latest-v-random.json
conda run -n zur_env python -m breakthrough_zero.cli arena \
  --agent-a neural --a-checkpoint "$LATEST8" --a-name latest-8x8 \
  --agent-b alphabeta --b-name alpha-beta \
  --board-size 8 --starting-rows 2 \
  --openings 30 --prefix-plies 6 --move-seconds 0.1 \
  --output results/phase6/latest-v-alphabeta.json
conda run -n zur_env python -m breakthrough_zero.cli arena \
  --agent-a neural --a-checkpoint "$LATEST8" --a-name latest-8x8 \
  --agent-b rollout --b-name rollout-puct \
  --board-size 8 --starting-rows 2 \
  --openings 30 --prefix-plies 6 --move-seconds 0.1 \
  --output results/phase6/latest-v-rollout.json
conda run -n zur_env python -m breakthrough_zero.cli arena \
  --agent-a neural --a-checkpoint "$LATEST8" --a-name latest-8x8 \
  --agent-b neural --b-checkpoint checkpoints/pretrained-8x8.keras --b-name pretrained-8x8 \
  --board-size 8 --starting-rows 2 \
  --openings 30 --prefix-plies 6 --move-seconds 0.1 \
  --output results/phase6/latest-v-pretrained.json
conda run -n zur_env python -m breakthrough_zero.cli elo-table \
  --reports results/phase6/latest-v-*.json \
  --anchor random \
  --output results/phase6/elo-table.json
