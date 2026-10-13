#!/usr/bin/env python3
"""Summarize the prespecified fixed-cohort K=1..50 locked-test analysis.

This is a post-lock secondary analysis. It reuses frozen per-user outputs and
does not refit or tune a model on the locked test cohort.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


KS = (1, 3, 5, 10, 20, 50)
PRIMARY = {
    "hierarchical_linear": ("0.03",),
    "lightgbm": ("intercept_speed", "0.03"),
    "residual_tcn": ("intercept_speed", "0.1"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence-index", required=True, type=Path)
    p.add_argument("--test-model-metrics", required=True, type=Path)
    p.add_argument("--hierarchical-metrics", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--bootstrap-replicates", type=int, default=10000)
    return p.parse_args()


def bootstrap(values: np.ndarray, rng: np.random.Generator, reps: int) -> list[float]:
    draws = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = read_rows(args.sequence_index)
    history_counts: dict[str, int] = {}
    for row in index:
        if row["role"] == "test_history":
            history_counts[row["user_key"]] = history_counts.get(row["user_key"], 0) + 1

    model_rows = read_rows(args.test_model_metrics)
    hl_rows = read_rows(args.hierarchical_metrics)
    users_at_50 = {
        row["user_key"] for row in model_rows
        if row["model"] == "lightgbm" and row["k"] == "50"
        and row["calibration"] == "intercept_speed" and row["ridge_alpha"] == "0.03"
    }
    if len(users_at_50) != 70:
        raise RuntimeError(f"Expected 70 users with K=50 metrics, found {len(users_at_50)}")

    selected: dict[tuple[str, int], dict[str, float]] = {}
    for k in KS:
        # The fixed cohort is applied to every K; all rows are paired by user.
        selected[("lightgbm", k)] = {
            row["user_key"]: float(row["mae_bpm"])
            for row in model_rows
            if row["model"] == "lightgbm" and int(row["k"]) == k
            and row["calibration"] == "intercept_speed" and row["ridge_alpha"] == "0.03"
            and row["user_key"] in users_at_50
        }
        selected[("residual_tcn", k)] = {
            row["user_key"]: float(row["mae_bpm"])
            for row in model_rows
            if row["model"] == "residual_tcn" and int(row["k"]) == k
            and row["calibration"] == "intercept_speed" and row["ridge_alpha"] == "0.1"
            and row["user_key"] in users_at_50
        }
        selected[("hierarchical_linear", k)] = {
            row["user_key"]: float(row["mae_bpm"])
            for row in hl_rows
            if int(row["k"]) == k and row["prior_precision_multiplier"] == "0.03"
            and row["user_key"] in users_at_50
        }
        for model in ("hierarchical_linear", "lightgbm", "residual_tcn"):
            if set(selected[(model, k)]) != users_at_50:
                raise RuntimeError(f"{model} K={k} does not contain the fixed 70-user cohort")

    rng = np.random.default_rng(20261012)
    records: list[dict[str, object]] = []
    for k in KS:
        baseline = selected[("hierarchical_linear", k)]
        for model in ("hierarchical_linear", "lightgbm", "residual_tcn"):
            values = np.asarray(list(selected[(model, k)].values()), dtype=float)
            improvement = np.asarray(
                [baseline[user] - selected[(model, k)][user] for user in sorted(users_at_50)],
                dtype=float,
            )
            mae_ci = bootstrap(values, rng, args.bootstrap_replicates)
            improvement_ci = bootstrap(improvement, rng, args.bootstrap_replicates)
            records.append({
                "model": model,
                "k": k,
                "users": len(values),
                "mean_mae_bpm": float(values.mean()),
                "mean_mae_bootstrap_95ci_low": mae_ci[0],
                "mean_mae_bootstrap_95ci_high": mae_ci[1],
                "improvement_vs_hierarchical_bpm": float(improvement.mean()),
                "improvement_vs_hierarchical_bootstrap_95ci_low": improvement_ci[0],
                "improvement_vs_hierarchical_bootstrap_95ci_high": improvement_ci[1],
                "proportion_users_improved_vs_hierarchical": float(np.mean(improvement > 0)),
            })

    with (args.output_dir / "fixed_cohort_history_depth.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    payload = {
        "analysis": "Fixed-cohort history-depth experiment",
        "stage": "Post-lock secondary analysis using frozen per-user test outputs",
        "fixed_users": len(users_at_50),
        "history_requirement": "All users had a K=50 metric and the same target workout was retained for every K.",
        "k_values": KS,
        "models": list(PRIMARY),
        "results": records,
        "guardrail": "No model was refit or tuned on the locked test cohort.",
    }
    (args.output_dir / "fixed_cohort_history_depth.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    styles = {"hierarchical_linear": ("Hierarchical linear", "#333333"), "lightgbm": ("LightGBM", "#0072B2"), "residual_tcn": ("Residual TCN", "#D55E00")}
    for model, (label, colour) in styles.items():
        vals = [next(r["mean_mae_bpm"] for r in records if r["model"] == model and r["k"] == k) for k in KS]
        ax.plot(KS, vals, marker="o", linewidth=2, label=label, color=colour)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("Personal-history workouts (K)")
    ax.set_ylabel("Mean user-level MAE (bpm)")
    ax.set_title("Fixed-cohort learning curves (70 users, same target)")
    ax.grid(True, alpha=.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "fixed_cohort_learning_curves.png", dpi=220)
    fig.savefig(args.output_dir / "fixed_cohort_learning_curves.pdf")
    plt.close(fig)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
