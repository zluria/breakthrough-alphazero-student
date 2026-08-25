#!/bin/bash
#SBATCH --job-name=bt-az8-serious
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-06:00:00
#SBATCH --output=results/slurm/%x-%j.out

set -euo pipefail
module load anaconda
PROJECT_DIR="$(pwd)"
export PYTHONPATH="$PROJECT_DIR/src"

STARTING_CHECKPOINT=results/phase6/learn-8x8/checkpoints/iteration-0005.keras
if [ ! -f "$STARTING_CHECKPOINT" ]; then
  echo "Missing starting checkpoint: $STARTING_CHECKPOINT"
  exit 1
fi

mkdir -p results/slurm results/phase8
conda run -n zur_env python -m unittest discover -s tests -v
conda run -n zur_env python -m breakthrough_zero.cli learn \
  --config configs/serious-az-8x8.json \
  --run-dir results/phase8/serious-az-8x8 \
  --checkpoint "$STARTING_CHECKPOINT" \
  --report results/phase8/serious-az-8x8-report.json
