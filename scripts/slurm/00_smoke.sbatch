#!/bin/bash
#SBATCH --job-name=bt-smoke
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --output=results/slurm/%x-%j.out

set -euo pipefail
module load anaconda
PROJECT_DIR="$(pwd)"
export PYTHONPATH="$PROJECT_DIR/src"

mkdir -p results/slurm results/smoke data/raw
conda run -n zur_env python -m unittest discover -s tests -v
conda run -n zur_env python -m breakthrough_zero.cli pretrain-data \
  --config configs/smoke-pretrain.json \
  --output data/raw/smoke-pretraining.jsonl.gz \
  --report results/smoke/pretraining-report.json
conda run -n zur_env python -m breakthrough_zero.cli inspect-data \
  --data data/raw/smoke-pretraining.jsonl.gz
