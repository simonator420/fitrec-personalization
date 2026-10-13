#!/usr/bin/env python3
"""Fit a nonlinear population model and validation-only personal calibration."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

from fit_baseline_partial_pooling import K_VALUES, linear_predict


FEATURE_NAMES = (
    "speed_kmh", "speed_sq", "speed_lag30", "speed_lag60",
    "cumulative_speed", "elapsed_minutes", "log_elapsed",
    "altitude_delta_m", "grade_30s", "vertical_rate_30s",
    "day_of_year_sin", "day_of_year_cos",
)
RIDGE_GRID = (0.0, .01, .03, .1, .3, 1.0, 3.0, 10.0, 30.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessed-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--coverage-threshold", type=float, default=.80)
    parser.add_argument("--max-points-per-workout", type=int, default=48)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    return parser.parse_args()


class Store:
    def __init__(self, root: Path, rows: list[dict[str, str]]):
        self.root = root
        self.rows = {row["workout_key"]: row for row in rows}
        self.cache_name: str | None = None
        self.cache = None

    def get(self, key: str) -> tuple[np.ndarray, ...]:
        row = self.rows[key]
        if row["shard"] != self.cache_name:
            with np.load(self.root / row["shard"]) as archive:
                self.cache = {
                    name: archive[name]
                    for name in (
                        "elapsed_seconds", "speed_kmh", "heart_rate_bpm", "altitude_m"
                    )
                }
            self.cache_name = row["shard"]
        start = int(row["offset"])
        end = start + int(row["length"])
        return tuple(
            self.cache[name][start:end].astype(float)
            for name in ("elapsed_seconds", "speed_kmh", "heart_rate_bpm", "altitude_m")
        )


def lag(values: np.ndarray, steps: int) -> np.ndarray:
    result = np.empty_like(values)
    result[:steps] = values[0]
    result[steps:] = values[:-steps]
    return result


def cumulative_mean(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    sums = np.cumsum(np.where(valid, values, 0.0))
    counts = np.cumsum(valid)
    return np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)


def first_finite_center(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    return values - finite[0] if len(finite) else values.copy()


def features(
    elapsed: np.ndarray, speed: np.ndarray, altitude: np.ndarray, timestamp: str
) -> np.ndarray:
    lag30 = lag(speed, 3)
    lag60 = lag(speed, 6)
    altitude30 = lag(altitude, 3)
    delta_altitude = altitude - altitude30
    horizontal_30 = np.maximum((lag30 + speed) / 2.0 / 3.6 * 30.0, 5.0)
    grade = np.clip(delta_altitude / horizontal_30, -.30, .30)
    vertical_rate = np.clip(delta_altitude / 30.0, -3.0, 3.0)
    moment = datetime.fromisoformat(timestamp)
    day = moment.timetuple().tm_yday
    return np.column_stack([
        speed,
        speed ** 2,
        lag30,
        lag60,
        cumulative_mean(speed),
        elapsed / 60.0,
        np.log1p(elapsed / 60.0),
        first_finite_center(altitude),
        grade,
        vertical_rate,
        np.full(len(speed), np.sin(2 * np.pi * day / 365.25)),
        np.full(len(speed), np.cos(2 * np.pi * day / 365.25)),
    ]).astype(np.float32)


def core_valid(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Altitude-derived columns may be NaN and are handled natively by LightGBM.
    required = [0, 1, 2, 3, 4, 5, 6, 10, 11]
    return np.isfinite(y) & np.all(np.isfinite(X[:, required]), axis=1)


def sample_indices(valid: np.ndarray, maximum: int) -> np.ndarray:
    indices = np.flatnonzero(valid)
    if len(indices) <= maximum:
        return indices
    positions = np.linspace(0, len(indices) - 1, maximum)
    return indices[np.round(positions).astype(int)]


def personal_stats(
    X: np.ndarray, y: np.ndarray, prediction: np.ndarray, kind: str
) -> tuple[np.ndarray, np.ndarray]:
    residual = y - prediction
    if kind == "intercept":
        adjustment = np.ones((len(y), 1))
    elif kind == "intercept_speed":
        adjustment = np.column_stack([np.ones(len(y)), X[:, 0] / 10.0])
    else:
        raise ValueError(kind)
    weight = 1.0 / len(y)
    return adjustment.T @ adjustment * weight, adjustment.T @ residual * weight


def solve_ridge(xx: np.ndarray, xy: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(xx.shape[0]) * alpha
    return np.linalg.solve(xx + penalty + np.eye(xx.shape[0]) * 1e-8, xy)


def bootstrap_ci(values: np.ndarray, replicates: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(replicates, len(values)), replace=True).mean(1)
    return [float(x) for x in np.quantile(means, [.025, .975])]


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_rows = list(csv.DictReader((args.preprocessed_dir / "sequence_index.csv").open()))
    if any(row["split"] == "test" for row in index_rows):
        raise RuntimeError("Locked test rows detected; model-development script refuses to continue")
    rows = [
        row for row in index_rows
        if float(row["hr_coverage_resampled"]) >= args.coverage_threshold
        and float(row["speed_coverage_resampled"]) >= args.coverage_threshold
    ]
    store = Store(args.preprocessed_dir, rows)
    development = sorted(
        [row for row in rows if row["role"] == "development_population"],
        key=lambda row: (row["shard"], int(row["offset"])),
    )
    X_parts, y_parts, weight_parts = [], [], []
    for row_index, row in enumerate(development, start=1):
        elapsed, speed, hr, altitude = store.get(row["workout_key"])
        X = features(elapsed, speed, altitude, row["timestamp_utc"])
        selected = sample_indices(core_valid(X, hr), args.max_points_per_workout)
        if not len(selected):
            continue
        X_parts.append(X[selected])
        y_parts.append(hr[selected].astype(np.float32))
        weight_parts.append(np.full(len(selected), 1.0 / len(selected), np.float32))
        if row_index % 5000 == 0:
            print(f"[features] development workouts={row_index:,}/{len(development):,}", flush=True)
    X_train = np.concatenate(X_parts)
    y_train = np.concatenate(y_parts)
    weights_train = np.concatenate(weight_parts)

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=500,
        learning_rate=.04,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=200,
        subsample=.85,
        colsample_bytree=.85,
        reg_lambda=1.0,
        random_state=20261008,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        X_train, y_train, sample_weight=weights_train,
        feature_name=list(FEATURE_NAMES),
    )
    joblib.dump(model, args.output_dir / "lightgbm_population.joblib", compress=3)

    validation = [row for row in rows if row["split"] == "validation"]
    sequences = {}
    for row in sorted(validation, key=lambda item: (item["shard"], int(item["offset"]))):
        elapsed, speed, hr, altitude = store.get(row["workout_key"])
        X = features(elapsed, speed, altitude, row["timestamp_utc"])
        valid = core_valid(X, hr)
        sequences[row["workout_key"]] = (
            X[valid], hr[valid], model.booster_.predict(X[valid])
        )
    by_user: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in validation:
        by_user[row["user_key"]].append(row)

    output: list[dict[str, object]] = []
    for user, user_rows in by_user.items():
        targets = [row for row in user_rows if row["role"] == "validation_target"]
        if len(targets) != 1:
            continue
        target = targets[0]
        histories = sorted(
            [row for row in user_rows if row["role"] == "validation_history"],
            key=lambda row: int(row["chronological_position"]),
        )
        X_target, y_target, pop_target = sequences[target["workout_key"]]
        if not len(y_target):
            continue
        target_adjustments = {
            "intercept": np.ones((len(y_target), 1)),
            "intercept_speed": np.column_stack([
                np.ones(len(y_target)), X_target[:, 0] / 10.0
            ]),
        }
        population_error = pop_target - y_target
        output.append({
            "user_key": user, "k": 0, "calibration": "population",
            "ridge_alpha": math.nan, "valid_points": len(y_target),
            "mae_bpm": float(np.mean(np.abs(population_error))),
            "rmse_bpm": float(np.sqrt(np.mean(population_error ** 2))),
            "bias_bpm": float(np.mean(population_error)),
        })
        history_stats: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
            "intercept": [], "intercept_speed": [],
        }
        for history in histories:
            X_h, y_h, pop_h = sequences[history["workout_key"]]
            if not len(y_h):
                continue
            for kind in history_stats:
                history_stats[kind].append(personal_stats(X_h, y_h, pop_h, kind))
        for k in K_VALUES[1:]:
            if any(len(values) < k for values in history_stats.values()):
                continue
            for kind, values in history_stats.items():
                selected = values[-k:]
                xx = sum((item[0] for item in selected), start=np.zeros_like(selected[0][0]))
                xy = sum((item[1] for item in selected), start=np.zeros_like(selected[0][1]))
                for alpha in RIDGE_GRID:
                    delta = solve_ridge(xx, xy, alpha)
                    prediction = pop_target + linear_predict(target_adjustments[kind], delta)
                    error = prediction - y_target
                    output.append({
                        "user_key": user, "k": k, "calibration": kind,
                        "ridge_alpha": alpha, "valid_points": len(y_target),
                        "mae_bpm": float(np.mean(np.abs(error))),
                        "rmse_bpm": float(np.sqrt(np.mean(error ** 2))),
                        "bias_bpm": float(np.mean(error)),
                    })

    with (args.output_dir / "validation_user_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)

    population = {
        row["user_key"]: float(row["mae_bpm"])
        for row in output if row["calibration"] == "population"
    }
    grouped: dict[tuple[int, str, float], list[dict[str, object]]] = defaultdict(list)
    for row in output:
        if row["calibration"] != "population":
            grouped[(
                int(row["k"]), str(row["calibration"]), float(row["ridge_alpha"])
            )].append(row)
    summaries = []
    for (k, calibration, alpha), group in sorted(grouped.items()):
        mae = np.asarray([float(row["mae_bpm"]) for row in group])
        benefit = np.asarray([
            population[str(row["user_key"])] - float(row["mae_bpm"])
            for row in group
        ])
        summaries.append({
            "k": k, "calibration": calibration, "ridge_alpha": alpha,
            "users": len(group), "mean_mae_bpm": float(mae.mean()),
            "median_mae_bpm": float(np.median(mae)),
            "mean_rmse_bpm": float(np.mean([float(row["rmse_bpm"]) for row in group])),
            "mean_bias_bpm": float(np.mean([float(row["bias_bpm"]) for row in group])),
            "mean_benefit_bpm": float(benefit.mean()),
            "benefit_q10_bpm": float(np.quantile(benefit, .10)),
            "proportion_users_improved": float(np.mean(benefit > 0)),
            "mean_mae_bootstrap_95ci": bootstrap_ci(
                mae, args.bootstrap_replicates,
                20261008 + k * 100 + int(alpha * 10),
            ),
            "mean_benefit_bootstrap_95ci": bootstrap_ci(
                benefit, args.bootstrap_replicates,
                20261009 + k * 100 + int(alpha * 10),
            ),
        })
    population_mae = np.asarray(list(population.values()))
    payload = {
        "stage": "Development fit and validation selection; locked test absent",
        "coverage_threshold": args.coverage_threshold,
        "training_workouts": len(development),
        "training_sampled_points": len(y_train),
        "max_points_per_workout": args.max_points_per_workout,
        "feature_names": FEATURE_NAMES,
        "lightgbm_parameters": model.get_params(),
        "validation_population": {
            "users": len(population),
            "mean_mae_bpm": float(population_mae.mean()),
            "median_mae_bpm": float(np.median(population_mae)),
            "mean_mae_bootstrap_95ci": bootstrap_ci(
                population_mae, args.bootstrap_replicates, 20261008
            ),
        },
        "personalization_grid": summaries,
        "guardrail": "Validation model-development output only; locked test absent.",
    }
    (args.output_dir / "lightgbm_validation_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
