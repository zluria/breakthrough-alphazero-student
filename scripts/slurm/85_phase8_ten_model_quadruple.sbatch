#!/bin/bash
#SBATCH --job-name=bt-az8-10model
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/%x-%j.out

set -euo pipefail
module load anaconda
PROJECT_DIR="$(pwd)"
export PYTHONPATH="$PROJECT_DIR/src"

RUN_DIR=results/phase8/final-elo-tournaments
CHECKPOINT_DIR="${CHECKPOINT_DIR:-results/phase8/serious-az-8x8/checkpoints}"
OUTPUT_DIR="$RUN_DIR/ten-model-quadruple-$SLURM_JOB_ID"
GAME_DIR="$OUTPUT_DIR/games"

iterations=(0015 0026 0036 0046 0056 0066 0076 0086 0096 0107)
names=(iteration-15 iteration-26 iteration-36 iteration-46 iteration-56 iteration-66 iteration-76 iteration-86 iteration-96 iteration-107)

for iteration in "${iterations[@]}"; do
  test -f "$CHECKPOINT_DIR/iteration-$iteration.keras"
done
mkdir -p results/slurm "$GAME_DIR"

echo "Source commit: $(git rev-parse HEAD)"
echo "Checkpoint directory: $CHECKPOINT_DIR"
conda run -n zur_env python -m unittest discover -s tests -v

# Every pair receives two fresh six-ply openings, each played with the colors
# reversed. Evaluation has no root noise or move sampling. A large time limit
# leaves the fixed count of 256 PUCT simulations in control of every move.
for ((i = 0; i < ${#iterations[@]}; i++)); do
  for ((j = i + 1; j < ${#iterations[@]}; j++)); do
    conda run -n zur_env python -m breakthrough_zero.cli arena \
      --agent-a neural --a-name "${names[$i]}" \
      --a-checkpoint "$CHECKPOINT_DIR/iteration-${iterations[$i]}.keras" \
      --agent-b neural --b-name "${names[$j]}" \
      --b-checkpoint "$CHECKPOINT_DIR/iteration-${iterations[$j]}.keras" \
      --board-size 8 --starting-rows 2 \
      --openings 2 --prefix-plies 6 \
      --move-seconds 1000 --simulations 256 \
      --output "$GAME_DIR/${names[$i]}-v-${names[$j]}.json"
  done
done

reports=("$GAME_DIR"/*.json)
test "${#reports[@]}" -eq 45
conda run -n zur_env python scripts/calculate_tournament_elos.py \
  --passes 300 --ranking "$OUTPUT_DIR/ranking.txt" "$GAME_DIR" \
  > "$OUTPUT_DIR/elo-300.json"
test "$(wc -l < "$OUTPUT_DIR/ranking.txt")" -eq 10
cat "$OUTPUT_DIR/elo-300.json"
