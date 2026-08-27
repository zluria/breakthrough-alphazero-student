#!/bin/bash
#SBATCH --job-name=bt-az8-top4
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

test -n "${FIRST_JOB_ID:-}"
RUN_DIR=results/phase8/final-elo-tournaments
CHECKPOINT_DIR="${CHECKPOINT_DIR:-results/phase8/serious-az-8x8/checkpoints}"
FIRST_OUTPUT_DIR="$RUN_DIR/ten-model-quadruple-$FIRST_JOB_ID"
OUTPUT_DIR="$RUN_DIR/top-four-tenfold-$SLURM_JOB_ID"
GAME_DIR="$OUTPUT_DIR/games"

mapfile -t ranking < "$FIRST_OUTPUT_DIR/ranking.txt"
top_names=("${ranking[@]:0:4}")
test "${#top_names[@]}" -eq 4
mkdir -p results/slurm "$GAME_DIR"
printf "%s\n" "${top_names[@]}" > "$OUTPUT_DIR/models.txt"

checkpoint_for_name() {
  number="${1#iteration-}"
  printf "%s/iteration-%04d.keras" "$CHECKPOINT_DIR" "$number"
}

for name in "${top_names[@]}"; do
  test -f "$(checkpoint_for_name "$name")"
done

echo "Source commit: $(git rev-parse HEAD)"
echo "Stage-one job: $FIRST_JOB_ID"
echo "Finalists: ${top_names[*]}"

# The finalists play ten games per pairing: five fresh six-ply openings with
# colors reversed. The stronger final uses exactly 1,024 simulations per move.
for ((i = 0; i < ${#top_names[@]}; i++)); do
  for ((j = i + 1; j < ${#top_names[@]}; j++)); do
    name_a="${top_names[$i]}"
    name_b="${top_names[$j]}"
    conda run -n zur_env python -m breakthrough_zero.cli arena \
      --agent-a neural --a-name "$name_a" \
      --a-checkpoint "$(checkpoint_for_name "$name_a")" \
      --agent-b neural --b-name "$name_b" \
      --b-checkpoint "$(checkpoint_for_name "$name_b")" \
      --board-size 8 --starting-rows 2 \
      --openings 5 --prefix-plies 6 \
      --move-seconds 1000 --simulations 1024 \
      --output "$GAME_DIR/$name_a-v-$name_b.json"
  done
done

reports=("$GAME_DIR"/*.json)
test "${#reports[@]}" -eq 6
conda run -n zur_env python scripts/calculate_tournament_elos.py \
  --passes 300 "$GAME_DIR" > "$OUTPUT_DIR/elo-300.json"
conda run -n zur_env python scripts/calculate_tournament_elos.py \
  --passes 300 "$FIRST_OUTPUT_DIR/games" "$GAME_DIR" \
  > "$OUTPUT_DIR/combined-elo-300.json"
cat "$OUTPUT_DIR/elo-300.json"
cat "$OUTPUT_DIR/combined-elo-300.json"
