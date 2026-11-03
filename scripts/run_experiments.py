"""
Run every (condition, seed) combination, each as its own process.

Runs that already have a metrics.json are skipped, so if a Colab session dies halfway
through I can just start the same command again and it picks up where it stopped.

    # main experiment: all four conditions, seeds 1 to 5
    python scripts/run_experiments.py --settings=configs/shakespeare_char.py --output=runs/main --seeds 1 2 3 4 5

    # learning-rate sweep for one condition, one seed
    python scripts/run_experiments.py --settings=configs/shakespeare_char.py --output=runs/lr_sweep \
        --conditions baseline --seeds 0 --sweep learning_rate=3e-4,6e-4,1e-3,2e-3

Any other --name=value arguments are passed straight to ablation.train for every run.
"""

import argparse
import itertools
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from ablation.settings import CONDITIONS  # noqa: E402


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", required=True, help="settings file, e.g. configs/shakespeare_char.py")
    parser.add_argument("--output", required=True, help="directory that gets one folder per run")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--sweep", default=None, help="name=value1,value2,... to try several values of one setting")
    return parser.parse_known_args()


def main():
    arguments, passed_through = parse_arguments()

    sweep_name, sweep_values = None, [None]
    if arguments.sweep:
        sweep_name, values = arguments.sweep.split("=", 1)
        sweep_values = values.split(",")

    for condition, sweep_value, seed in itertools.product(arguments.conditions, sweep_values, arguments.seeds):
        run_name = condition if sweep_value is None else f"{condition}_{sweep_name}{sweep_value}"
        run_directory = os.path.join(arguments.output, f"{run_name}_s{seed}")
        if os.path.exists(os.path.join(run_directory, "metrics.json")):
            print(f"skipping {run_directory}, already done")
            continue

        command = [
            sys.executable, "-m", "ablation.train",
            f"--settings={arguments.settings}",
            f"--condition={condition}",
            f"--seed={seed}",
            f"--output_directory={run_directory}",
            *passed_through,
        ]
        if sweep_value is not None:
            command.append(f"--{sweep_name}={sweep_value}")
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
