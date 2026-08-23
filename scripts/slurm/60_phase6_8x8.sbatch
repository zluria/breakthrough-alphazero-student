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
  --epochs 8 --batch-size 128 --seed 20260812
conda run -n zur_env python -m breakthrough_zero.cli learn \
  --config configs/learn-8x8.json \
  --run-dir results/phase6/learn-8x8 \
  --checkpoint checkpoints/pretrained-8x8.keras \
  --report results/phase6/learn-8x8-report.json
