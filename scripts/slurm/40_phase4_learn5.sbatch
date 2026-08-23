#!/bin/bash
#SBATCH --job-name=bt-learn5
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/slurm/%x-%j.out

set -euo pipefail
module load anaconda
PROJECT_DIR="$(pwd)"
export PYTHONPATH="$PROJECT_DIR/src"

mkdir -p results/slurm results/phase4
conda run -n zur_env python -m unittest discover -s tests -v
conda run -n zur_env python -m breakthrough_zero.cli learn \
  --config configs/learn-5x5.json \
  --run-dir results/phase4/learn-5x5 \
  --checkpoint checkpoints/pretrained-5x5.keras \
  --report results/phase4/learn-5x5-report.json
