#!/bin/bash
#SBATCH --job-name=bt-az8-smoke
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=results/slurm/%x-%j.out

set -euo pipefail
module load anaconda
PROJECT_DIR="$(pwd)"
export PYTHONPATH="$PROJECT_DIR/src"

STARTING_CHECKPOINT=results/phase6/learn-8x8/checkpoints/iteration-0005.keras
mkdir -p results/slurm results/phase8
conda run -n zur_env python -m pytest
conda run -n zur_env python -m breakthrough_zero.cli learn \
  --config configs/smoke-serious-8x8.json \
  --run-dir "results/phase8/smoke-$SLURM_JOB_ID" \
  --checkpoint "$STARTING_CHECKPOINT" \
  --report "results/phase8/smoke-$SLURM_JOB_ID-report.json"
