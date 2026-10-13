#!/usr/bin/env python3
"""Run a frozen-hyperparameter feature-matched model ablation.

Population models are fit only on development workouts. The locked test is
used once for the requested K=5 secondary comparison; no test result is used
for tuning. The script intentionally keeps the two feature sets identical
between the hierarchical-linear and LightGBM implementations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np

from fit_baseline_partial_pooling import linear_predict, solve
from fit_lightgbm_personalization import (
    FEATURE_NAMES as EXTENDED_NAMES,
    Store,
    core_valid,
    features,
    sample_indices,
    solve_ridge,
)


BASE_NAMES = (
    "speed_kmh", "speed_sq", "speed_lag30", "speed_lag60",
    "cumulative_speed", "elapsed_minutes", "log_elapsed",
)
K = 5
ALPHA = 0.03


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--development-dir", required=True, type=Path)
    p.add_argument("--test-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--coverage-threshold", type=float, default=.80)
    p.add_argument("--max-points-per-workout", type=int, default=48)
    p.add_argument("--bootstrap-replicates", type=int, default=10000)
    return p.parse_args()


def read_index(root: Path) -> list[dict[str, str]]:
    with (root / "sequence_index.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def extended_or_base(elapsed: np.ndarray, speed: np.ndarray, altitude: np.ndarray, timestamp: str, kind: str) -> np.ndarray:
    # Match the validated speed-cap preprocessing used for the current model
    # ladder and keep malformed legacy values from contaminating linear fits.
    elapsed = np.clip(np.nan_to_num(elapsed, nan=0.0, posinf=86400.0, neginf=0.0), 0.0, 86400.0)
    speed = np.clip(np.nan_to_num(speed, nan=0.0, posinf=55.0, neginf=0.0), 0.0, 55.0)
    altitude = np.clip(np.nan_to_num(altitude, nan=0.0, posinf=10000.0, neginf=-10000.0), -10000.0, 10000.0)
    X_all = np.nan_to_num(features(elapsed, speed, altitude, timestamp), nan=0.0, posinf=0.0, neginf=0.0)
    if kind == "base":
        X = X_all[:, :7]
    elif kind == "extended":
        X = X_all
    else:
        raise ValueError(kind)
    # Keep all linear algebra in a well-scaled finite range. This only affects
    # malformed legacy points; the validated workout values are far below it.
    return np.clip(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), -1e4, 1e4).astype(np.float64)


def valid(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.isfinite(y) & (y >= 30.0) & (y <= 250.0) & np.all(np.isfinite(X), axis=1)


def benchmark_valid(elapsed: np.ndarray, speed: np.ndarray, altitude: np.ndarray, timestamp: str, y: np.ndarray) -> np.ndarray:
    """Use the same original-point validity rule as the frozen benchmark."""
    elapsed = np.clip(np.asarray(elapsed, dtype=np.float64), 0.0, 86400.0)
    speed = np.clip(np.asarray(speed, dtype=np.float64), 0.0, 55.0)
    altitude = np.clip(np.asarray(altitude, dtype=np.float64), -10000.0, 10000.0)
    return core_valid(features(elapsed, speed, altitude, timestamp), y)


def bootstrap(values: np.ndarray, seed: int, reps: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [.025, .975])]


def train_linear(kind: str, dev_rows: list[dict[str, str]], store: Store) -> np.ndarray:
    xtx = None
    xty = None
    for row in dev_rows:
        elapsed, speed, hr, altitude = store.get(row["workout_key"])
        X = extended_or_base(elapsed, speed, altitude, row["timestamp_utc"], kind)
        mask = benchmark_valid(elapsed, speed, altitude, row["timestamp_utc"], hr)
        if not mask.any():
            continue
        Xv = np.column_stack([np.ones(mask.sum()), X[mask]]).astype(np.float64)
        yv = np.clip(np.asarray(hr[mask], dtype=np.float64), 30.0, 250.0)
        n = len(yv)
        if not n:
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            xx = Xv.T @ Xv / n
            xy = Xv.T @ yv / n
        if not np.isfinite(xx).all() or not np.isfinite(xy).all():
            continue
        if xtx is None:
            xtx, xty = xx, xy
        else:
            xtx += xx
            xty += xy
    if xtx is None:
        raise RuntimeError(f"No valid development data for {kind}")
    return solve(xtx, xty, ridge=1e-4)


def train_lightgbm(kind: str, dev_rows: list[dict[str, str]], store: Store, maximum: int) -> lgb.LGBMRegressor:
    X_parts, y_parts, weights = [], [], []
    for row in dev_rows:
        elapsed, speed, hr, altitude = store.get(row["workout_key"])
        X = extended_or_base(elapsed, speed, altitude, row["timestamp_utc"], kind)
        mask = benchmark_valid(elapsed, speed, altitude, row["timestamp_utc"], hr)
        selected = sample_indices(mask, maximum)
        if len(selected):
            X_parts.append(X[selected].astype(np.float32))
            y_parts.append(hr[selected].astype(np.float32))
            weights.append(np.full(len(selected), 1.0 / len(selected), np.float32))
    X_train = np.concatenate(X_parts)
    y_train = np.concatenate(y_parts)
    weight = np.concatenate(weights)
    model = lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=500, learning_rate=.04,
        num_leaves=31, max_depth=-1, min_child_samples=200,
        subsample=.85, colsample_bytree=.85, reg_lambda=1.0,
        random_state=20261012, n_jobs=-1, verbosity=-1,
    )
    names = BASE_NAMES if kind == "base" else EXTENDED_NAMES
    model.fit(X_train, y_train, sample_weight=weight, feature_name=list(names))
    return model


def calibration(history: list[tuple[np.ndarray, np.ndarray]], predictions: list[np.ndarray], target_X: np.ndarray, target_pop: np.ndarray, target_y: np.ndarray, ridge: float) -> float:
    xx = np.zeros((2, 2))
    xy = np.zeros(2)
    for X, residual in zip(history, predictions):
        A = np.column_stack([np.ones(len(residual)), X[:, 0] / 10.0])
        xx += A.T @ A / len(residual)
        xy += A.T @ residual / len(residual)
    delta = solve_ridge(xx, xy, ridge)
    A_target = np.column_stack([np.ones(len(target_y)), target_X[:, 0] / 10.0])
    return float(np.mean(np.abs(target_pop + A_target @ delta - target_y)))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dev_rows = read_index(args.development_dir)
    test_rows = read_index(args.test_dir)
    dev_rows = [r for r in dev_rows if r["role"] == "development_population" and float(r["hr_coverage_resampled"]) >= args.coverage_threshold and float(r["speed_coverage_resampled"]) >= args.coverage_threshold]
    test_rows = [r for r in test_rows if float(r["hr_coverage_resampled"]) >= args.coverage_threshold and float(r["speed_coverage_resampled"]) >= args.coverage_threshold]
    dev_store = Store(args.development_dir, dev_rows)
    test_store = Store(args.test_dir, test_rows)
    models: dict[tuple[str, str], object] = {}
    for kind in ("base", "extended"):
        print(f"[fit] hierarchical-linear {kind}", flush=True)
        models[("hierarchical_linear", kind)] = train_linear(kind, dev_rows, dev_store)
        print(f"[fit] lightgbm {kind}", flush=True)
        models[("lightgbm", kind)] = train_lightgbm(kind, dev_rows, dev_store, args.max_points_per_workout)

    by_user: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in test_rows:
        by_user[row["user_key"]].append(row)
    output: list[dict[str, object]] = []
    for user, rows in sorted(by_user.items()):
        targets = [r for r in rows if r["role"] == "test_target"]
        histories = sorted([r for r in rows if r["role"] == "test_history"], key=lambda r: int(r["chronological_position"]))
        if len(targets) != 1 or len(histories) < K:
            continue
        target = targets[0]
        histories = histories[-K:]
        for kind in ("base", "extended"):
            target_raw = test_store.get(target["workout_key"])
            elapsed_t, speed_t, hr_t, altitude_t = target_raw
            X_t = extended_or_base(elapsed_t, speed_t, altitude_t, target["timestamp_utc"], kind)
            mask_t = benchmark_valid(elapsed_t, speed_t, altitude_t, target["timestamp_utc"], hr_t)
            X_t, y_t = X_t[mask_t], hr_t[mask_t]
            if not len(y_t):
                continue
            for model_name in ("hierarchical_linear", "lightgbm"):
                model = models[(model_name, kind)]
                if model_name == "hierarchical_linear":
                    pop_t = np.column_stack([np.ones(len(X_t)), X_t]) @ model
                else:
                    pop_t = model.predict(X_t)
                hist_X, hist_res = [], []
                for history in histories:
                    elapsed_h, speed_h, hr_h, altitude_h = test_store.get(history["workout_key"])
                    X_h = extended_or_base(elapsed_h, speed_h, altitude_h, history["timestamp_utc"], kind)
                    mask_h = benchmark_valid(elapsed_h, speed_h, altitude_h, history["timestamp_utc"], hr_h)
                    X_h, y_h = X_h[mask_h], hr_h[mask_h]
                    if not len(y_h):
                        continue
                    if model_name == "hierarchical_linear":
                        pop_h = np.column_stack([np.ones(len(X_h)), X_h]) @ model
                    else:
                        pop_h = model.predict(X_h)
                    hist_X.append(X_h)
                    hist_res.append(y_h - pop_h)
                if len(hist_X) < K:
                    continue
                mae = calibration(hist_X, hist_res, X_t, pop_t, y_t, ALPHA)
                output.append({"user_key": user, "target_workout_key": target["workout_key"], "model": model_name, "feature_set": kind, "k": K, "mae_bpm": mae, "valid_points": len(y_t)})

    if not output:
        raise RuntimeError("No eligible test users")
    expected_cells = {
        ("hierarchical_linear", "base"),
        ("hierarchical_linear", "extended"),
        ("lightgbm", "base"),
        ("lightgbm", "extended"),
    }
    by_user_cells: dict[str, set[tuple[str, str]]] = defaultdict(set)
    by_user_targets: dict[str, set[str]] = defaultdict(set)
    for row in output:
        user = str(row["user_key"])
        by_user_cells[user].add((str(row["model"]), str(row["feature_set"])))
        by_user_targets[user].add(str(row["target_workout_key"]))
    incomplete = {u: sorted(cells) for u, cells in by_user_cells.items() if cells != expected_cells}
    inconsistent_targets = {u: sorted(targets) for u, targets in by_user_targets.items() if len(targets) != 1}
    if incomplete or inconsistent_targets:
        raise RuntimeError(
            "Feature-matched guardrail failed: all four arms must contain the same users "
            "and exactly one shared target workout per user; "
            f"incomplete={list(incomplete)[:5]}, inconsistent_targets={list(inconsistent_targets)[:5]}"
        )
    with (args.output_dir / "feature_matched_ablation_user_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in output:
        groups[(str(row["model"]), str(row["feature_set"]))].append(row)
    by_key = {(str(r["user_key"]), str(r["model"]), str(r["feature_set"])): float(r["mae_bpm"]) for r in output}
    summary = []
    for (model, kind), rows in sorted(groups.items()):
        values = np.asarray([float(r["mae_bpm"]) for r in rows])
        base = np.asarray([by_key[(str(r["user_key"]), model, "base")] for r in rows])
        diff = base - values
        summary.append({"model": model, "feature_set": kind, "users": len(values), "mean_mae_bpm": float(values.mean()), "mean_mae_bootstrap_95ci": bootstrap(values, 20261020 + len(summary), args.bootstrap_replicates), "improvement_vs_base_bpm": float(diff.mean()), "improvement_vs_base_bootstrap_95ci": bootstrap(diff, 20261040 + len(summary), args.bootstrap_replicates)})
    for kind in ("base", "extended"):
        h = {(str(r["user_key"])): float(r["mae_bpm"]) for r in groups[("hierarchical_linear", kind)]}
        l = {(str(r["user_key"])): float(r["mae_bpm"]) for r in groups[("lightgbm", kind)]}
        diff = np.asarray([h[u] - l[u] for u in sorted(set(h) & set(l))])
        summary.append({"comparison": "lightgbm_vs_hierarchical_linear", "feature_set": kind, "users": len(diff), "improvement_bpm": float(diff.mean()), "improvement_bootstrap_95ci": bootstrap(diff, 20261060 + len(summary), args.bootstrap_replicates)})
    payload = {"analysis": "Feature-matched model ablation", "stage": "Post-lock secondary analysis; all population fitting used development data only", "k": K, "coverage_threshold": args.coverage_threshold, "personalization_ridge_alpha": ALPHA, "feature_sets": {"base": list(BASE_NAMES), "extended": list(EXTENDED_NAMES)}, "summary": summary, "guardrail": "No test result was used for feature selection, tuning, or model choice."}
    (args.output_dir / "feature_matched_ablation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (args.output_dir / "feature_matched_ablation_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in summary for key in row}))
        writer.writeheader(); writer.writerows(summary)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    labels, vals = [], []
    for row in summary:
        if "mean_mae_bpm" in row:
            labels.append(f"{row['model']}\n{row['feature_set']}"); vals.append(row["mean_mae_bpm"])
    ax.bar(labels, vals, color=["#999999", "#0072B2", "#BBBBBB", "#D55E00"])
    ax.set_ylabel("Mean user-level MAE (bpm)"); ax.set_title("Feature-matched model ablation (K=5)"); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(args.output_dir / "feature_matched_ablation.png", dpi=220); fig.savefig(args.output_dir / "feature_matched_ablation.pdf"); plt.close(fig)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
