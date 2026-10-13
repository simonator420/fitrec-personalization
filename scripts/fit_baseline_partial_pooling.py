#!/usr/bin/env python3
"""Fit a transparent development-only population model and validate pooling.

The target-workout HR trajectory is never used for fitting. Validation history
HR is used only to estimate the validation user's personal adjustment. Locked
test data are absent from the preprocessed directory by construction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


K_VALUES = (0, 1, 3, 5, 10, 20, 50)
FEATURE_NAMES = (
    "intercept",
    "speed_div10",
    "speed_sq_div100",
    "speed_lag30_div10",
    "speed_lag60_div10",
    "cumulative_mean_speed_div10",
    "elapsed_hours",
    "log_elapsed_div5",
    "speed_x_elapsed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessed-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--coverage-threshold", type=float, default=0.80)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    return parser.parse_args()


def lag(values: np.ndarray, steps: int) -> np.ndarray:
    output = np.empty_like(values, dtype=float)
    output[:steps] = values[0]
    output[steps:] = values[:-steps]
    return output


def cumulative_nanmean(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    sums = np.cumsum(np.where(finite, values, 0.0))
    counts = np.cumsum(finite)
    return np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)


def design_matrix(elapsed: np.ndarray, speed: np.ndarray) -> np.ndarray:
    speed10 = speed / 10.0
    elapsed_hours = elapsed / 3600.0
    return np.column_stack([
        np.ones(len(speed)),
        speed10,
        speed10 ** 2,
        lag(speed, 3) / 10.0,
        lag(speed, 6) / 10.0,
        cumulative_nanmean(speed) / 10.0,
        elapsed_hours,
        np.log1p(elapsed / 60.0) / 5.0,
        speed10 * elapsed_hours,
    ])


def linear_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Return row-wise linear predictions without platform-BLAS matmul noise."""
    if X.ndim != 2 or beta.ndim != 1 or X.shape[1] != len(beta):
        raise ValueError("Incompatible design matrix and coefficient vector")
    return np.sum(X * beta[None, :], axis=1)


def weighted_stats(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int]:
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Xv = X[valid]
    yv = y[valid]
    if not len(yv):
        p = X.shape[1]
        return np.zeros((p, p)), np.zeros(p), 0.0, 0
    weight = 1.0 / len(yv)
    return Xv.T @ Xv * weight, Xv.T @ yv * weight, float(yv @ yv * weight), len(yv)


def personal_stats(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int]:
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Xv = X[valid]
    yv = y[valid]
    if not len(yv):
        return np.zeros((2, 2)), np.zeros(2), 0.0, 0
    residual = yv - linear_predict(Xv, beta)
    adjustment = np.column_stack([np.ones(len(yv)), Xv[:, 1]])
    weight = 1.0 / len(yv)
    return (
        adjustment.T @ adjustment * weight,
        adjustment.T @ residual * weight,
        float(residual @ residual * weight),
        len(yv),
    )


def solve(matrix: np.ndarray, vector: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    penalty = np.eye(matrix.shape[0]) * ridge
    penalty[0, 0] = ridge * 0.01
    return np.linalg.solve(matrix + penalty, vector)


def bootstrap_mean_interval(values: np.ndarray, replicates: int) -> list[float | None]:
    if not len(values):
        return [None, None]
    rng = np.random.default_rng(20261005)
    means = np.empty(replicates)
    for index in range(replicates):
        means[index] = np.mean(rng.choice(values, size=len(values), replace=True))
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


class SequenceStore:
    def __init__(self, root: Path, rows: list[dict[str, str]]):
        self.root = root
        self.rows = {row["workout_key"]: row for row in rows}
        self.cache_name: str | None = None
        self.cache = None

    def get(self, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        row = self.rows[key]
        shard = row["shard"]
        if shard != self.cache_name:
            self.cache = np.load(self.root / shard)
            self.cache_name = shard
        start = int(row["offset"])
        end = start + int(row["length"])
        return (
            self.cache["elapsed_seconds"][start:end].astype(float),
            self.cache["speed_kmh"][start:end].astype(float),
            self.cache["heart_rate_bpm"][start:end].astype(float),
        )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((args.preprocessed_dir / "sequence_index.csv").open()))
    if any(row["split"] == "test" for row in rows):
        raise RuntimeError("Locked test rows detected; development script refuses to continue")
    eligible = {
        row["workout_key"] for row in rows
        if float(row["hr_coverage_resampled"]) >= args.coverage_threshold
        and float(row["speed_coverage_resampled"]) >= args.coverage_threshold
    }
    rows = [row for row in rows if row["workout_key"] in eligible]
    store = SequenceStore(args.preprocessed_dir, rows)

    p = len(FEATURE_NAMES)
    xtx = np.zeros((p, p))
    xty = np.zeros(p)
    development_rows = sorted(
        [row for row in rows if row["role"] == "development_population"],
        key=lambda row: (row["shard"], int(row["offset"])),
    )
    for row in development_rows:
        elapsed, speed, hr = store.get(row["workout_key"])
        X = design_matrix(elapsed, speed)
        s_xx, s_xy, _, _ = weighted_stats(X, hr)
        xtx += s_xx
        xty += s_xy
    beta = solve(xtx, xty, ridge=1e-4)

    user_accumulators: dict[str, dict[str, object]] = defaultdict(
        lambda: {"xx": np.zeros((2, 2)), "xy": np.zeros(2), "workouts": 0}
    )
    residual_sse = 0.0
    residual_workouts = 0
    for row in development_rows:
        elapsed, speed, hr = store.get(row["workout_key"])
        X = design_matrix(elapsed, speed)
        u_xx, u_xy, u_sse, n = personal_stats(X, hr, beta)
        if n:
            accumulator = user_accumulators[row["user_key"]]
            accumulator["xx"] += u_xx
            accumulator["xy"] += u_xy
            accumulator["workouts"] += 1
            residual_sse += u_sse
            residual_workouts += 1
    development_deltas = []
    for accumulator in user_accumulators.values():
        if accumulator["workouts"] >= 5 and np.linalg.matrix_rank(accumulator["xx"]) == 2:
            development_deltas.append(solve(accumulator["xx"], accumulator["xy"], ridge=1e-3))

    deltas = np.asarray(development_deltas)
    prior_mean = np.mean(deltas, axis=0)
    prior_cov = np.cov(deltas.T)
    prior_cov += np.eye(2) * 1e-3
    prior_precision = np.linalg.inv(prior_cov)
    sigma2 = residual_sse / residual_workouts

    validation_rows = [row for row in rows if row["split"] == "validation"]
    validation_sequences = {}
    for row in sorted(validation_rows, key=lambda item: (item["shard"], int(item["offset"]))):
        validation_sequences[row["workout_key"]] = store.get(row["workout_key"])
    by_val_user: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in validation_rows:
        by_val_user[row["user_key"]].append(row)
    metrics: list[dict[str, object]] = []

    for user, user_rows in by_val_user.items():
        targets = [row for row in user_rows if row["role"] == "validation_target"]
        if len(targets) != 1:
            continue
        target = targets[0]
        histories = sorted(
            [row for row in user_rows if row["role"] == "validation_history"],
            key=lambda row: int(row["chronological_position"]),
        )
        elapsed_t, speed_t, hr_t = validation_sequences[target["workout_key"]]
        X_t = design_matrix(elapsed_t, speed_t)
        valid_t = np.isfinite(hr_t) & np.all(np.isfinite(X_t), axis=1)
        X_eval = X_t[valid_t]
        y_eval = hr_t[valid_t]
        if not len(y_eval):
            continue
        adjustment_t = np.column_stack([np.ones(len(X_eval)), X_eval[:, 1]])
        population_prediction = linear_predict(X_eval, beta)

        for k in K_VALUES:
            if len(histories) < k:
                continue
            selected = histories[-k:] if k else []
            s_xx = np.zeros((2, 2))
            s_xy = np.zeros(2)
            for history in selected:
                elapsed_h, speed_h, hr_h = validation_sequences[history["workout_key"]]
                X_h = design_matrix(elapsed_h, speed_h)
                h_xx, h_xy, _, n = personal_stats(X_h, hr_h, beta)
                if n:
                    s_xx += h_xx
                    s_xy += h_xy

            if k == 0:
                partial_delta = np.zeros(2)
                no_pool_delta = np.zeros(2)
            else:
                partial_delta = np.linalg.solve(
                    s_xx / sigma2 + prior_precision,
                    s_xy / sigma2 + prior_precision @ prior_mean,
                )
                no_pool_delta = solve(s_xx, s_xy, ridge=1e-3)

            predictions = {
                "population": population_prediction,
                "partial_pooling": population_prediction + adjustment_t @ partial_delta,
                "no_pooling": population_prediction + adjustment_t @ no_pool_delta,
            }
            for model, prediction in predictions.items():
                error = prediction - y_eval
                metrics.append({
                    "user_key": user,
                    "target_workout_key": target["workout_key"],
                    "k": k,
                    "model": model,
                    "valid_points": len(y_eval),
                    "mae_bpm": float(np.mean(np.abs(error))),
                    "rmse_bpm": float(np.sqrt(np.mean(error ** 2))),
                    "bias_bpm": float(np.mean(error)),
                })

    with (args.output_dir / "validation_user_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]) if metrics else [])
        if metrics:
            writer.writeheader()
            writer.writerows(metrics)

    grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in metrics:
        grouped[(int(row["k"]), str(row["model"]))].append(row)
    summaries = []
    for (k, model), group in sorted(grouped.items()):
        mae = np.asarray([float(row["mae_bpm"]) for row in group])
        rmse = np.asarray([float(row["rmse_bpm"]) for row in group])
        bias = np.asarray([float(row["bias_bpm"]) for row in group])
        summaries.append({
            "k": k,
            "model": model,
            "users": len(group),
            "mean_mae_bpm": float(np.mean(mae)),
            "median_mae_bpm": float(np.median(mae)),
            "mean_rmse_bpm": float(np.mean(rmse)),
            "mean_bias_bpm": float(np.mean(bias)),
            "mae_mean_bootstrap_95ci": bootstrap_mean_interval(mae, args.bootstrap_replicates),
        })

    benefits = []
    for k in K_VALUES:
        pop = {
            row["user_key"]: float(row["mae_bpm"])
            for row in metrics if row["k"] == k and row["model"] == "population"
        }
        for model in ("partial_pooling", "no_pooling"):
            candidate = {
                row["user_key"]: float(row["mae_bpm"])
                for row in metrics if row["k"] == k and row["model"] == model
            }
            users = sorted(set(pop) & set(candidate))
            benefit = np.asarray([pop[user] - candidate[user] for user in users])
            if not len(benefit):
                continue
            benefits.append({
                "k": k,
                "comparison": f"population_minus_{model}",
                "users": len(users),
                "mean_benefit_bpm": float(np.mean(benefit)),
                "median_benefit_bpm": float(np.median(benefit)),
                "proportion_users_improved": float(np.mean(benefit > 0)),
                "mean_benefit_bootstrap_95ci": bootstrap_mean_interval(benefit, args.bootstrap_replicates),
            })

    model = {
        "stage": "Development/validation benchmark; locked test absent",
        "coverage_threshold": args.coverage_threshold,
        "feature_names": FEATURE_NAMES,
        "population_coefficients": beta.tolist(),
        "personal_adjustment": ["intercept", "speed_div10"],
        "prior_mean": prior_mean.tolist(),
        "prior_covariance": prior_cov.tolist(),
        "residual_variance_effective_workout_weighting": sigma2,
        "development_workouts": len(development_rows),
        "development_users": len(user_accumulators),
        "validation_summary": summaries,
        "personalization_benefit": benefits,
        "guardrail": "Exploratory validation benchmark only; do not report as locked test performance.",
    }
    (args.output_dir / "baseline_model_and_validation.json").write_text(
        json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(model, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
