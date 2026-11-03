"""
Summarise a folder of runs and compare each trick with the baseline.

    python scripts/analyze_results.py runs/main --plot paper/figures/main.png

For each condition it prints the mean and standard deviation of the final validation
loss over seeds. For each trick against the baseline it prints:

  - the difference in mean loss (negative means the trick did better),
  - a 95% confidence interval for that difference,
  - Welch's t-test p-value, which doesn't assume both groups vary by the same amount,
  - the Holm-corrected p-value, because three comparisons are made at once and
    some would look "significant" by luck if each were tested on its own.

Runs are grouped by folder name with the trailing _s<seed> removed, so learning-rate
sweep folders like baseline_learning_rate1e-3_s0 form their own groups.
"""

import argparse
import glob
import json
import math
import os
import re

import numpy as np
from scipy import stats

CONDITION_ORDER = ["baseline", "rope", "rmsnorm", "muon"]
CONDITION_COLORS = {"baseline": "#2a78d6", "rope": "#eb6834", "rmsnorm": "#1baf7a", "muon": "#eda100"}
TEXT_COLOR, SECONDARY_TEXT_COLOR, GRID_COLOR = "#0b0b0b", "#52514e", "#e4e3df"


def load_runs(root_directory):
    groups = {}
    for metrics_path in sorted(glob.glob(os.path.join(root_directory, "*", "metrics.json"))):
        with open(metrics_path) as metrics_file:
            run = json.load(metrics_file)
        folder_name = os.path.basename(os.path.dirname(metrics_path))
        group_name = re.sub(r"_s\d+$", "", folder_name)
        groups.setdefault(group_name, []).append(run)
    return groups


def display_order(group_name):
    if group_name in CONDITION_ORDER:
        return (CONDITION_ORDER.index(group_name), group_name)
    return (len(CONDITION_ORDER), group_name)


def finished_runs(runs):
    return [run for run in runs if not run["diverged"]]


def welch_comparison(trick_losses, baseline_losses):
    """Difference in means (trick - baseline), its 95% confidence interval, degrees of freedom and p-value."""
    trick_losses = np.asarray(trick_losses)
    baseline_losses = np.asarray(baseline_losses)
    trick_variance_of_mean = trick_losses.var(ddof=1) / len(trick_losses)
    baseline_variance_of_mean = baseline_losses.var(ddof=1) / len(baseline_losses)
    standard_error = math.sqrt(trick_variance_of_mean + baseline_variance_of_mean)

    # Welch-Satterthwaite degrees of freedom.
    degrees_of_freedom = (trick_variance_of_mean + baseline_variance_of_mean) ** 2 / (
        trick_variance_of_mean**2 / (len(trick_losses) - 1)
        + baseline_variance_of_mean**2 / (len(baseline_losses) - 1)
    )
    difference = trick_losses.mean() - baseline_losses.mean()
    margin = stats.t.ppf(0.975, degrees_of_freedom) * standard_error
    p_value = stats.ttest_ind(trick_losses, baseline_losses, equal_var=False).pvalue
    return difference, difference - margin, difference + margin, degrees_of_freedom, p_value


def holm_correction(p_values):
    """
    Holm-Bonferroni adjusted p-values, returned in the same order as given.
    The smallest p-value is multiplied by the number of tests, the next one by one less,
    and so on, never letting an adjusted value drop below the one before it.
    """
    number_of_tests = len(p_values)
    adjusted = np.empty(number_of_tests)
    largest_so_far = 0.0
    for rank, index in enumerate(np.argsort(p_values)):
        largest_so_far = max(largest_so_far, (number_of_tests - rank) * p_values[index])
        adjusted[index] = min(1.0, largest_so_far)
    return adjusted


def summarise_groups(groups):
    print("| condition | runs | diverged | final val loss (mean ± sd) | perplexity | ms/step |")
    print("|---|---|---|---|---|---|")
    summary = {}
    for group_name in sorted(groups, key=display_order):
        runs = groups[group_name]
        good_runs = finished_runs(runs)
        losses = np.array([run["final_val_loss"] for run in good_runs])
        milliseconds_per_step = np.mean([run["ms_per_step"] for run in good_runs]) if good_runs else float("nan")
        standard_deviation = losses.std(ddof=1) if len(losses) > 1 else float("nan")
        summary[group_name] = {
            "runs": len(good_runs),
            "diverged": len(runs) - len(good_runs),
            "mean": float(losses.mean()) if len(losses) else None,
            "sd": float(standard_deviation),
            "ms_per_step": float(milliseconds_per_step),
            "values": losses.tolist(),
        }
        if len(losses):
            mean_text = f"{losses.mean():.4f} ± {standard_deviation:.4f}"
            perplexity_text = f"{math.exp(losses.mean()):.3f}"
        else:
            mean_text = perplexity_text = "-"
        print(
            f"| {group_name} | {len(runs)} | {len(runs) - len(good_runs)} | {mean_text}"
            f" | {perplexity_text} | {milliseconds_per_step:.1f} |"
        )
    return summary


def compare_with_baseline(summary):
    tricks = [name for name in sorted(summary, key=display_order) if name in CONDITION_ORDER and name != "baseline"]
    if "baseline" not in summary or not tricks:
        return
    baseline_losses = summary["baseline"]["values"]
    comparisons = {name: welch_comparison(summary[name]["values"], baseline_losses) for name in tricks}
    adjusted_p_values = holm_correction([comparisons[name][4] for name in tricks])

    print("\n| condition | difference vs baseline | 95% CI | Welch df | p | p (Holm) |")
    print("|---|---|---|---|---|---|")
    for name, adjusted_p in zip(tricks, adjusted_p_values):
        difference, lower, upper, degrees_of_freedom, p_value = comparisons[name]
        print(
            f"| {name} | {difference:+.4f} | [{lower:+.4f}, {upper:+.4f}] | {degrees_of_freedom:.1f}"
            f" | {p_value:.3g} | {adjusted_p:.3g} |"
        )
        summary[name].update(
            {
                "difference": difference,
                "confidence_interval": [lower, upper],
                "degrees_of_freedom": degrees_of_freedom,
                "p": p_value,
                "p_holm": float(adjusted_p),
            }
        )


def plot(groups, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [name for name in sorted(groups, key=display_order) if name in CONDITION_COLORS]
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": GRID_COLOR,
            "axes.labelcolor": SECONDARY_TEXT_COLOR,
            "xtick.color": SECONDARY_TEXT_COLOR,
            "ytick.color": SECONDARY_TEXT_COLOR,
            "text.color": TEXT_COLOR,
        }
    )
    figure, (curve_axis, final_axis) = plt.subplots(
        1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.6, 1]}
    )

    # Left: validation loss during training, averaged over seeds.
    lowest_loss = float("inf")
    for name in names:
        runs = finished_runs(groups[name])
        steps = [point["step"] for point in runs[0]["history"]]
        curves = np.array([[point["val_loss"] for point in run["history"]] for run in runs])
        lowest_loss = min(lowest_loss, curves.min())
        # The baseline is dashed and drawn on top so it stays visible when another line overlaps it.
        is_baseline = name == "baseline"
        curve_axis.plot(
            steps,
            curves.mean(axis=0),
            color=CONDITION_COLORS[name],
            linewidth=1.5,
            linestyle="--" if is_baseline else "-",
            zorder=3 if is_baseline else 2,
            label=name,
        )
    curve_axis.set_xlabel("training step")
    curve_axis.set_ylabel("validation loss (fixed batches)")
    curve_axis.set_title("(a) Validation loss during training, mean over seeds", fontsize=9, loc="left")
    curve_axis.set_ylim(lowest_loss - 0.05, lowest_loss + 1.0)
    curve_axis.legend(frameon=False, fontsize=8)

    # Right: final full-split validation loss, one dot per seed and a line at the mean.
    for position, name in enumerate(names):
        losses = [run["final_val_loss"] for run in finished_runs(groups[name])]
        offsets = np.linspace(-0.12, 0.12, len(losses)) if len(losses) > 1 else [0.0]
        final_axis.scatter(
            position + np.asarray(offsets),
            losses,
            s=22,
            color=CONDITION_COLORS[name],
            edgecolors="white",
            linewidths=1,
            zorder=3,
        )
        final_axis.hlines(np.mean(losses), position - 0.25, position + 0.25, color=TEXT_COLOR, linewidth=1.2, zorder=4)
    final_axis.set_xticks(range(len(names)), names)
    final_axis.set_ylabel("final validation loss (full split)")
    final_axis.set_title("(b) Final loss per seed (line = mean)", fontsize=9, loc="left")

    for axis in (curve_axis, final_axis):
        axis.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    figure.savefig(output_path, dpi=200)
    print(f"wrote {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_directory")
    parser.add_argument("--plot", default=None, help="where to save the figure, e.g. paper/figures/main.png")
    parser.add_argument(
        "--baseline-from",
        default=None,
        help="take the baseline runs from another folder (used for the matched-time Muon runs)",
    )
    arguments = parser.parse_args()

    groups = load_runs(arguments.runs_directory)
    if arguments.baseline_from:
        groups["baseline"] = load_runs(arguments.baseline_from)["baseline"]
    if not groups:
        raise SystemExit(f"no metrics.json files found under {arguments.runs_directory}")

    summary = summarise_groups(groups)
    compare_with_baseline(summary)
    with open(os.path.join(arguments.runs_directory, "summary.json"), "w") as summary_file:
        json.dump(summary, summary_file, indent=1)
    if arguments.plot:
        plot(groups, arguments.plot)


if __name__ == "__main__":
    main()
