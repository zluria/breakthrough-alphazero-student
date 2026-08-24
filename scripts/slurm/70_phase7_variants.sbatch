#!/bin/bash
#SBATCH --job-name=bt-variants
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/slurm/%x-%j.out

set -euo pipefail
module load anaconda
PROJECT_DIR="$(pwd)"
export PYTHONPATH="$PROJECT_DIR/src"

mkdir -p results/slurm results/phase7 checkpoints
conda run -n zur_env python -m breakthrough_zero.cli pretrain-network \
  --data data/raw/pretraining-5x5-10000.jsonl.gz \
  --output checkpoints/pretrained-small-5x5.keras \
  --report results/phase7/pretrained-small-report.json \
  --epochs 8 --batch-size 128 \
  --filters 24 --residual-blocks 2
conda run -n zur_env python -m breakthrough_zero.cli learn \
  --config configs/research-baseline-5x5.json \
  --run-dir results/phase7/baseline \
  --checkpoint checkpoints/pretrained-5x5.keras \
  --report results/phase7/baseline-report.json
conda run -n zur_env python -m breakthrough_zero.cli learn \
  --config configs/research-no-noise-5x5.json \
  --run-dir results/phase7/no-noise \
  --checkpoint checkpoints/pretrained-5x5.keras \
  --report results/phase7/no-noise-report.json
conda run -n zur_env python -m breakthrough_zero.cli learn \
  --config configs/research-small-5x5.json \
  --run-dir results/phase7/small \
  --checkpoint checkpoints/pretrained-small-5x5.keras \
  --report results/phase7/small-report.json
conda run -n zur_env python -m breakthrough_zero.cli arena \
  --agent-a neural --a-checkpoint results/phase7/baseline/checkpoints/iteration-0004.keras --a-name baseline \
  --agent-b neural --b-checkpoint results/phase7/no-noise/checkpoints/iteration-0004.keras --b-name no-noise \
  --board-size 5 --starting-rows 1 --openings 50 --prefix-plies 4 --move-seconds 0.1 \
  --output results/phase7/baseline-v-no-noise.json
conda run -n zur_env python -m breakthrough_zero.cli arena \
  --agent-a neural --a-checkpoint results/phase7/baseline/checkpoints/iteration-0004.keras --a-name baseline \
  --agent-b neural --b-checkpoint results/phase7/small/checkpoints/iteration-0004.keras --b-name small \
  --board-size 5 --starting-rows 1 --openings 50 --prefix-plies 4 --move-seconds 0.1 \
  --output results/phase7/baseline-v-small.json
