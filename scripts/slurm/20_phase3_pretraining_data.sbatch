#!/bin/bash
#SBATCH --job-name=bt-phase3-data
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/slurm/%x-%j.out

set -euo pipefail
module load anaconda
PROJECT_DIR="$(pwd)"
export PYTHONPATH="$PROJECT_DIR/src"

mkdir -p results/slurm results/phase3 data/raw
conda run -n zur_env python -m unittest discover -s tests -v
conda run -n zur_env python -m breakthrough_zero.cli pretrain-data \
  --config configs/pretrain-5x5.json \
  --output data/raw/pretraining-5x5-10000.jsonl.gz \
  --report results/phase3/pretraining-data-report.json
