#!/bin/bash
#SBATCH --job-name=bt-pretrain-net
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/%x-%j.out

set -euo pipefail
module load anaconda
PROJECT_DIR="$(pwd)"
export PYTHONPATH="$PROJECT_DIR/src"

mkdir -p results/slurm results/phase4 checkpoints
conda run -n zur_env python -m unittest discover -s tests -v
conda run -n zur_env python -m breakthrough_zero.cli pretrain-network \
  --data data/raw/pretraining-5x5-10000.jsonl.gz \
  --output checkpoints/pretrained-5x5.keras \
  --report results/phase4/pretraining-network-report.json \
  --epochs 8 \
  --batch-size 128 \
  --seed 20260811
