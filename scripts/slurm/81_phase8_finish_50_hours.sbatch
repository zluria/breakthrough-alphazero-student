#!/bin/bash
#SBATCH --job-name=bt-az8-finish
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/slurm/%x-%j.out

set -euo pipefail
module load anaconda
PROJECT_DIR="$(pwd)"
export PYTHONPATH="$PROJECT_DIR/src"

PREVIOUS_PROJECT=/home/zurlu/breakthrough-alphazero-e8d08ba
mkdir -p results/slurm results/phase8
conda run -n zur_env python -m unittest discover -s tests -v
conda run -n zur_env python scripts/continue_serious_run.py \
  "$PREVIOUS_PROJECT" \
  results/phase8/serious-az-8x8-continuation \
  results/phase8/serious-az-8x8-continuation-report.json
