#!/bin/bash
#SBATCH --job-name=bt-phase2
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/%x-%j.out

set -euo pipefail
module load anaconda
PROJECT_DIR="$(pwd)"
export PYTHONPATH="$PROJECT_DIR/src"

mkdir -p results/slurm results/phase2
conda run -n zur_env python -m breakthrough_zero.cli diagnostic \
  --output-dir results/phase2 \
  --examples 512 \
  --epochs 12 \
  --batch-size 64 \
  --seed 20260811
