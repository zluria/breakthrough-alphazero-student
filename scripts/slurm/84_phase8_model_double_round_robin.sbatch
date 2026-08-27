#!/bin/bash
#SBATCH --job-name=bt-az8-model-rr
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/%x-%j.out

set -euo pipefail
module load anaconda
PROJECT_DIR="$(pwd)"
export PYTHONPATH="$PROJECT_DIR/src"

RUN_DIR=results/phase8/serious-az-8x8
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
OUTPUT_DIR="$RUN_DIR/model-double-round-robin-$SLURM_JOB_ID"

iterations=(0005 0015 0026 0036 0046 0056 0066 0076)
names=(start iteration-15 iteration-26 iteration-36 iteration-46 iteration-56 iteration-66 iteration-76)

for iteration in "${iterations[@]}"; do
  test -f "$CHECKPOINT_DIR/iteration-$iteration.keras"
done
mkdir -p results/slurm "$OUTPUT_DIR"

conda run -n zur_env python -m unittest discover -s tests -v

# Every pair receives one fresh six-ply opening and plays it with both color
# assignments. The generous time ceiling never controls play: every move stops
# after exactly 64 PUCT simulations, matching the scheduled strength checks.
for ((i = 0; i < ${#iterations[@]}; i++)); do
  for ((j = i + 1; j < ${#iterations[@]}; j++)); do
    conda run -n zur_env python -m breakthrough_zero.cli arena \
      --agent-a neural --a-name "${names[$i]}" \
      --a-checkpoint "$CHECKPOINT_DIR/iteration-${iterations[$i]}.keras" \
      --agent-b neural --b-name "${names[$j]}" \
      --b-checkpoint "$CHECKPOINT_DIR/iteration-${iterations[$j]}.keras" \
      --board-size 8 --starting-rows 2 \
      --openings 1 --prefix-plies 6 \
      --move-seconds 1000 --simulations 64 \
      --output "$OUTPUT_DIR/${names[$i]}-v-${names[$j]}.json"
  done
done
