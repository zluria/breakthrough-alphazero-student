#!/bin/bash
#SBATCH --job-name=bt-az8-robin
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/%x-%j.out

set -euo pipefail
module load anaconda
PROJECT_DIR="$(pwd)"
export PYTHONPATH="$PROJECT_DIR/src"

RUN_DIR=results/phase8/serious-az-8x8
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
OUTPUT_DIR="$RUN_DIR/round-robin-$SLURM_JOB_ID"

for iteration in 0015 0026 0036 0046; do
  test -f "$CHECKPOINT_DIR/iteration-$iteration.keras"
done
mkdir -p results/slurm "$OUTPUT_DIR"

# The six agents play four games against every opponent. Two fresh random
# six-ply openings are each played twice with the agents' colors reversed.
# Search noise and move sampling are absent. The large simulation ceiling makes
# the common 0.1-second move allowance the effective search limit.
run_pair() {
  conda run -n zur_env python -m breakthrough_zero.cli arena \
    "$@" \
    --board-size 8 --starting-rows 2 \
    --openings 2 --prefix-plies 6 \
    --move-seconds 0.1 --simulations 1000000
}

conda run -n zur_env python -m unittest discover -s tests -v

run_pair \
  --agent-a rollout --a-name rollout-puct \
  --agent-b alphabeta --b-name alpha-beta \
  --output "$OUTPUT_DIR/rollout-v-alphabeta.json"

run_pair \
  --agent-a rollout --a-name rollout-puct \
  --agent-b neural --b-name iteration-15 \
  --b-checkpoint "$CHECKPOINT_DIR/iteration-0015.keras" \
  --output "$OUTPUT_DIR/rollout-v-iteration-15.json"
run_pair \
  --agent-a rollout --a-name rollout-puct \
  --agent-b neural --b-name iteration-26 \
  --b-checkpoint "$CHECKPOINT_DIR/iteration-0026.keras" \
  --output "$OUTPUT_DIR/rollout-v-iteration-26.json"
run_pair \
  --agent-a rollout --a-name rollout-puct \
  --agent-b neural --b-name iteration-36 \
  --b-checkpoint "$CHECKPOINT_DIR/iteration-0036.keras" \
  --output "$OUTPUT_DIR/rollout-v-iteration-36.json"
run_pair \
  --agent-a rollout --a-name rollout-puct \
  --agent-b neural --b-name iteration-46 \
  --b-checkpoint "$CHECKPOINT_DIR/iteration-0046.keras" \
  --output "$OUTPUT_DIR/rollout-v-iteration-46.json"

run_pair \
  --agent-a alphabeta --a-name alpha-beta \
  --agent-b neural --b-name iteration-15 \
  --b-checkpoint "$CHECKPOINT_DIR/iteration-0015.keras" \
  --output "$OUTPUT_DIR/alphabeta-v-iteration-15.json"
run_pair \
  --agent-a alphabeta --a-name alpha-beta \
  --agent-b neural --b-name iteration-26 \
  --b-checkpoint "$CHECKPOINT_DIR/iteration-0026.keras" \
  --output "$OUTPUT_DIR/alphabeta-v-iteration-26.json"
run_pair \
  --agent-a alphabeta --a-name alpha-beta \
  --agent-b neural --b-name iteration-36 \
  --b-checkpoint "$CHECKPOINT_DIR/iteration-0036.keras" \
  --output "$OUTPUT_DIR/alphabeta-v-iteration-36.json"
run_pair \
  --agent-a alphabeta --a-name alpha-beta \
  --agent-b neural --b-name iteration-46 \
  --b-checkpoint "$CHECKPOINT_DIR/iteration-0046.keras" \
  --output "$OUTPUT_DIR/alphabeta-v-iteration-46.json"

run_pair \
  --agent-a neural --a-name iteration-15 \
  --a-checkpoint "$CHECKPOINT_DIR/iteration-0015.keras" \
  --agent-b neural --b-name iteration-26 \
  --b-checkpoint "$CHECKPOINT_DIR/iteration-0026.keras" \
  --output "$OUTPUT_DIR/iteration-15-v-iteration-26.json"
run_pair \
  --agent-a neural --a-name iteration-15 \
  --a-checkpoint "$CHECKPOINT_DIR/iteration-0015.keras" \
  --agent-b neural --b-name iteration-36 \
  --b-checkpoint "$CHECKPOINT_DIR/iteration-0036.keras" \
  --output "$OUTPUT_DIR/iteration-15-v-iteration-36.json"
run_pair \
  --agent-a neural --a-name iteration-15 \
  --a-checkpoint "$CHECKPOINT_DIR/iteration-0015.keras" \
  --agent-b neural --b-name iteration-46 \
  --b-checkpoint "$CHECKPOINT_DIR/iteration-0046.keras" \
  --output "$OUTPUT_DIR/iteration-15-v-iteration-46.json"
run_pair \
  --agent-a neural --a-name iteration-26 \
  --a-checkpoint "$CHECKPOINT_DIR/iteration-0026.keras" \
  --agent-b neural --b-name iteration-36 \
  --b-checkpoint "$CHECKPOINT_DIR/iteration-0036.keras" \
  --output "$OUTPUT_DIR/iteration-26-v-iteration-36.json"
run_pair \
  --agent-a neural --a-name iteration-26 \
  --a-checkpoint "$CHECKPOINT_DIR/iteration-0026.keras" \
  --agent-b neural --b-name iteration-46 \
  --b-checkpoint "$CHECKPOINT_DIR/iteration-0046.keras" \
  --output "$OUTPUT_DIR/iteration-26-v-iteration-46.json"
run_pair \
  --agent-a neural --a-name iteration-36 \
  --a-checkpoint "$CHECKPOINT_DIR/iteration-0036.keras" \
  --agent-b neural --b-name iteration-46 \
  --b-checkpoint "$CHECKPOINT_DIR/iteration-0046.keras" \
  --output "$OUTPUT_DIR/iteration-36-v-iteration-46.json"
