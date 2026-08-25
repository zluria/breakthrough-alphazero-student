"""Continue the serious run if its first Slurm segment stopped early."""

import glob
import json
import os
import sys

from breakthrough_zero.training import run_learning_loop


def full_old_path(project_dir, path):
    if os.path.isabs(path):
        return path
    return os.path.join(project_dir, path)


def main():
    previous_project = sys.argv[1]
    output_dir = sys.argv[2]
    output_report = sys.argv[3]
    previous_report_path = os.path.join(
        previous_project,
        "results/phase8/serious-az-8x8-report.json",
    )
    with open(previous_report_path, "r", encoding="utf-8") as stream:
        previous = json.load(stream)

    remaining_hours = 50 - previous["elapsed_s"] / 3600
    if remaining_hours <= 0.25:
        print("The first segment used the full 50-hour budget.")
        return 0

    with open(
        "configs/serious-az-8x8.json",
        "r",
        encoding="utf-8",
    ) as stream:
        config = json.load(stream)

    config["max_hours"] = remaining_hours
    config["iterations"] = 200
    config["search_schedule"] = [
        {
            "until_iteration": 200,
            "full_simulations": 256,
            "fast_simulations": 32,
        }
    ]
    config.pop("max_strength_stalls", None)
    config["strength_reference_checkpoint"] = full_old_path(
        previous_project,
        previous["best_checkpoint"],
    )

    raw_pattern = os.path.join(
        previous_project,
        "results/phase8/serious-az-8x8/raw/iteration-*.jsonl.gz",
    )
    raw_paths = sorted(glob.glob(raw_pattern))
    if not raw_paths:
        raise FileNotFoundError("the first segment has no replay records")
    # Seven recent tranches slightly exceed the replay capacity. Loading them
    # in order therefore reconstructs the same recent FIFO window.
    config["replay_seed_paths"] = raw_paths[-7:]

    latest_checkpoint = full_old_path(
        previous_project,
        previous["latest_checkpoint"],
    )
    report = run_learning_loop(
        config,
        output_dir,
        latest_checkpoint,
    )
    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
