#!/usr/bin/env python3
"""Train and evaluate an adapted FitRec-style LSTM at K=5.

This is a transparent adaptation of the Ni et al. (WWW 2019) idea: an LSTM
encodes the recent workout history and a second LSTM encodes the target
workout context. It is *not* an exact reproduction of the original task or
preprocessing. Population fitting uses development users only; the locked
test is used once for the requested secondary comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fit_lightgbm_personalization import Store, core_valid, features as benchmark_features


K = 5
SEED = 20261012
HISTORY_DIM = 6
TARGET_DIM = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--development-dir", required=True, type=Path)
    p.add_argument("--test-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--sequence-length", type=int, default=128)
    p.add_argument("--hidden-history", type=int, default=32)
    p.add_argument("--hidden-target", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--coverage-threshold", type=float, default=.80)
    p.add_argument("--validation-fraction", type=float, default=.20)
    p.add_argument("--bootstrap-replicates", type=int, default=10000)
    p.add_argument("--reference-lightgbm", type=Path)
    p.add_argument("--reference-hierarchical", type=Path)
    p.add_argument("--reference-tcn", type=Path)
    return p.parse_args()


def read_index(root: Path) -> list[dict[str, str]]:
    with (root / "sequence_index.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_values(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=high, neginf=low), low, high)


def resample(values: np.ndarray, length: int) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(length, dtype=np.float32)
    if len(values) == length:
        return values.astype(np.float32)
    positions = np.linspace(0, len(values) - 1, length)
    return values[np.round(positions).astype(int)].astype(np.float32)


def resample_indices(count: int, length: int) -> np.ndarray:
    if count <= 0:
        return np.zeros(length, dtype=np.int64)
    positions = np.linspace(0, count - 1, length)
    return np.round(positions).astype(np.int64)


def raw_target_arrays(elapsed: np.ndarray, speed: np.ndarray, altitude: np.ndarray, hr: np.ndarray, timestamp: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_elapsed = np.clip(np.asarray(elapsed, dtype=np.float64), 0.0, 86400.0)
    raw_speed = np.clip(np.asarray(speed, dtype=np.float64), 0.0, 55.0)
    raw_altitude = np.clip(np.asarray(altitude, dtype=np.float64), -10000.0, 10000.0)
    benchmark_X = benchmark_features(raw_elapsed, raw_speed, raw_altitude, timestamp)
    raw_hr = np.asarray(hr, dtype=np.float64)
    valid_hr = core_valid(benchmark_X, raw_hr)
    elapsed = finite_values(raw_elapsed, 0.0, 86400.0)
    speed = finite_values(raw_speed, 0.0, 55.0)
    altitude = finite_values(raw_altitude, -10000.0, 10000.0)
    altitude -= altitude[0] if len(altitude) else 0.0
    cumulative_distance = np.cumsum(speed / 3.6 * 10.0)
    X = np.column_stack([
        speed / 55.0,
        np.clip(altitude / 500.0, -20.0, 20.0),
        elapsed / 3600.0,
        cumulative_distance / 10000.0,
    ])
    return X.astype(np.float32), raw_hr.astype(np.float32), valid_hr


def fixed_target_arrays(elapsed: np.ndarray, speed: np.ndarray, altitude: np.ndarray, hr: np.ndarray, timestamp: str, length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create fixed-length training inputs without selecting points by HR availability."""
    X, raw_hr, valid_hr = raw_target_arrays(elapsed, speed, altitude, hr, timestamp)
    indices = resample_indices(len(X), length)
    y = np.nan_to_num(raw_hr[indices], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mask = valid_hr[indices].astype(np.float32)
    return X[indices], y, mask


def history_summary(elapsed: np.ndarray, speed: np.ndarray, hr: np.ndarray, altitude: np.ndarray) -> np.ndarray:
    elapsed = finite_values(elapsed, 0.0, 86400.0)
    speed = finite_values(speed, 0.0, 55.0)
    raw_hr = np.asarray(hr, dtype=np.float64)
    valid_hr = raw_hr[np.isfinite(raw_hr) & (raw_hr >= 30.0) & (raw_hr <= 250.0)]
    hr = valid_hr if len(valid_hr) else np.asarray([100.0], dtype=np.float64)
    altitude = finite_values(altitude, -10000.0, 10000.0)
    gain = float(np.sum(np.maximum(np.diff(altitude), 0.0))) if len(altitude) > 1 else 0.0
    loss = float(np.sum(np.maximum(-np.diff(altitude), 0.0))) if len(altitude) > 1 else 0.0
    return np.asarray([
        float(np.nanmean(speed)) / 55.0,
        float(np.nanmean(hr)) / 200.0,
        float(np.nanstd(hr)) / 50.0,
        float(elapsed[-1] if len(elapsed) else 0.0) / 3600.0,
        float(np.sum(speed / 3.6 * 10.0)) / 10000.0,
        (gain + loss) / 1000.0,
    ], dtype=np.float32)


class WorkoutDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, examples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        history, target, y, mask = self.examples[index]
        return torch.from_numpy(history), torch.from_numpy(target), torch.from_numpy(y), torch.from_numpy(mask)


class AdaptedFitRecLSTM(nn.Module):
    def __init__(self, history_dim: int, target_dim: int, hidden_history: int, hidden_target: int):
        super().__init__()
        self.history_lstm = nn.LSTM(history_dim, hidden_history, batch_first=True)
        self.target_lstm = nn.LSTM(target_dim, hidden_target, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_history + hidden_target, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, history: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _, (history_hidden, _) = self.history_lstm(history)
        target_out, _ = self.target_lstm(target)
        history_context = history_hidden[-1].unsqueeze(1).expand(-1, target_out.shape[1], -1)
        return self.head(torch.cat([target_out, history_context], dim=-1)).squeeze(-1)


def make_examples(rows: list[dict[str, str]], store: Store, sequence_length: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    by_user: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_user[row["user_key"]].append(row)
    examples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for user_rows in by_user.values():
        ordered = sorted(user_rows, key=lambda r: int(r["chronological_position"]))
        for index in range(K, len(ordered)):
            history_rows = ordered[index - K:index]
            target_row = ordered[index]
            history = []
            for row in history_rows:
                elapsed, speed, hr, altitude = store.get(row["workout_key"])
                history.append(history_summary(elapsed, speed, hr, altitude))
            elapsed, speed, hr, altitude = store.get(target_row["workout_key"])
            target, y, mask = fixed_target_arrays(elapsed, speed, altitude, hr, target_row["timestamp_utc"], sequence_length)
            if mask.any() and np.isfinite(target).all() and np.isfinite(y).all():
                examples.append((np.asarray(history, dtype=np.float32), target, y, mask))
    return examples


def train_model(model: AdaptedFitRecLSTM, train: WorkoutDataset, holdout: WorkoutDataset, args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    torch.manual_seed(SEED)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    holdout_loader = DataLoader(holdout, batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.SmoothL1Loss(reduction="none")
    model.to(device)
    best_state = None
    best_holdout = math.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); train_losses = []
        for h, x, y, mask in loader:
            h, x, y, mask = h.to(device), x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            raw_loss = loss_fn(model(h, x), y)
            loss = (raw_loss * mask).sum() / mask.sum().clamp_min(1.0)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval(); holdout_losses = []
        with torch.no_grad():
            for h, x, y, mask in holdout_loader:
                raw_loss = loss_fn(model(h.to(device), x.to(device)), y.to(device))
                mask = mask.to(device)
                holdout_losses.append(float(((raw_loss * mask).sum() / mask.sum().clamp_min(1.0)).cpu()))
        train_loss = float(np.mean(train_losses)) if train_losses else math.nan
        holdout_loss = float(np.mean(holdout_losses)) if holdout_losses else math.nan
        history.append({"epoch": epoch, "train_loss": train_loss, "holdout_loss": holdout_loss})
        if np.isfinite(holdout_loss) and holdout_loss < best_holdout:
            best_holdout = holdout_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"epochs": args.epochs, "best_holdout_loss": best_holdout, "history": history}


def bootstrap(values: np.ndarray, seed: int, reps: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(draws, [.025, .975])]


def evaluate(model: AdaptedFitRecLSTM, rows: list[dict[str, str]], store: Store, sequence_length: int, device: torch.device) -> list[dict[str, object]]:
    by_user: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_user[row["user_key"]].append(row)
    model.eval(); output = []
    with torch.no_grad():
        for user, user_rows in sorted(by_user.items()):
            target_rows = [row for row in user_rows if row["role"] == "test_target"]
            histories = sorted([row for row in user_rows if row["role"] == "test_history"], key=lambda r: int(r["chronological_position"]))
            if len(target_rows) != 1 or len(histories) < K:
                continue
            history_features = []
            for row in histories[-K:]:
                elapsed, speed, hr, altitude = store.get(row["workout_key"])
                history_features.append(history_summary(elapsed, speed, hr, altitude))
            target_row = target_rows[0]
            elapsed, speed, hr, altitude = store.get(target_row["workout_key"])
            target, y, valid_hr = raw_target_arrays(elapsed, speed, altitude, hr, target_row["timestamp_utc"])
            if not valid_hr.any():
                continue
            h = torch.from_numpy(np.asarray(history_features, dtype=np.float32)).unsqueeze(0).to(device)
            x = torch.from_numpy(target).unsqueeze(0).to(device)
            prediction = model(h, x).squeeze(0).cpu().numpy()
            output.append({
                "user_key": user,
                "target_workout_key": target_row["workout_key"],
                "model": "adapted_fitrec_lstm",
                "k": K,
                "mae_bpm": float(np.mean(np.abs(prediction[valid_hr] - y[valid_hr]))),
                "valid_points": int(valid_hr.sum()),
            })
    return output


def load_reference(path: Path | None, model: str, calibration: str | None = None, ridge_alpha: str | None = None) -> tuple[dict[str, float], dict[str, str]]:
    if path is None:
        return {}, {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[str, float] = {}
    targets: dict[str, str] = {}
    for row in rows:
        if row.get("model") == model and row.get("k") == str(K):
            if calibration is not None and row.get("calibration") != calibration:
                continue
            if ridge_alpha is not None and row.get("ridge_alpha") != ridge_alpha:
                continue
            output[row["user_key"]] = float(row["mae_bpm"])
            if row.get("target_workout_key"):
                targets[row["user_key"]] = row["target_workout_key"]
        elif model == "hierarchical_linear" and row.get("k") == str(K) and row.get("prior_precision_multiplier") == "0.03":
            output[row["user_key"]] = float(row["mae_bpm"])
            if row.get("target_workout_key"):
                targets[row["user_key"]] = row["target_workout_key"]
    return output, targets


def split_development_users(rows: list[dict[str, str]], validation_fraction: float) -> tuple[list[dict[str, str]], list[dict[str, str]], set[str], set[str]]:
    """Deterministically split whole development users; no workout can cross the split."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1")
    users = np.asarray(sorted({row["user_key"] for row in rows}), dtype=object)
    if len(users) < 2:
        raise RuntimeError("At least two development users are required for user-disjoint validation")
    rng = np.random.default_rng(SEED)
    shuffled = users[rng.permutation(len(users))]
    validation_count = min(max(1, int(round(len(users) * validation_fraction))), len(users) - 1)
    validation_users = {str(user) for user in shuffled[:validation_count]}
    training_users = {str(user) for user in users if str(user) not in validation_users}
    training_rows = [row for row in rows if row["user_key"] in training_users]
    validation_rows = [row for row in rows if row["user_key"] in validation_users]
    return training_rows, validation_rows, training_users, validation_users


def main() -> int:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_rows = read_index(args.development_dir)
    test_rows = read_index(args.test_dir)
    dev_rows = [row for row in dev_rows if row["role"] == "development_population" and float(row["hr_coverage_resampled"]) >= args.coverage_threshold and float(row["speed_coverage_resampled"]) >= args.coverage_threshold]
    test_rows = [row for row in test_rows if float(row["hr_coverage_resampled"]) >= args.coverage_threshold and float(row["speed_coverage_resampled"]) >= args.coverage_threshold]
    dev_store = Store(args.development_dir, dev_rows)
    test_store = Store(args.test_dir, test_rows)
    train_rows, validation_rows, training_users, validation_users = split_development_users(dev_rows, args.validation_fraction)
    train_store = Store(args.development_dir, train_rows)
    validation_store = Store(args.development_dir, validation_rows)
    train_examples = make_examples(train_rows, train_store, args.sequence_length)
    validation_examples = make_examples(validation_rows, validation_store, args.sequence_length)
    if len(train_examples) < 20 or not validation_examples:
        raise RuntimeError(
            f"Insufficient user-disjoint development examples for K={K}: "
            f"train={len(train_examples)}, validation={len(validation_examples)}"
        )
    model = AdaptedFitRecLSTM(HISTORY_DIM, TARGET_DIM, args.hidden_history, args.hidden_target)
    training = train_model(model, WorkoutDataset(train_examples), WorkoutDataset(validation_examples), args, device)
    training.update({
        "training_users": len(training_users),
        "validation_users": len(validation_users),
        "user_overlap": len(training_users & validation_users),
        "validation_fraction": args.validation_fraction,
    })
    with (args.output_dir / "adapted_fitrec_user_split.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["user_key", "split"])
        writer.writeheader()
        writer.writerows([{"user_key": user, "split": "training"} for user in sorted(training_users)])
        writer.writerows([{"user_key": user, "split": "validation"} for user in sorted(validation_users)])
    torch.save({"state_dict": model.state_dict(), "config": vars(args), "adaptation": "Ni et al. FitRec-style history encoder plus target-workout LSTM; not exact original reproduction."}, args.output_dir / "adapted_fitrec_lstm_k5.pt")
    metrics = evaluate(model, test_rows, test_store, args.sequence_length, device)
    if not metrics:
        raise RuntimeError("No eligible test users for adapted FitRec evaluation")
    with (args.output_dir / "adapted_fitrec_lstm_user_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    values = np.asarray([float(row["mae_bpm"]) for row in metrics])
    target_map = {row["user_key"]: row["target_workout_key"] for row in metrics}
    if len(target_map) != len(metrics) or any(not row["target_workout_key"] for row in metrics):
        raise RuntimeError("Target-workout guardrail failed: each evaluated user must have exactly one target workout")
    summary = {"model": "adapted_fitrec_lstm", "k": K, "users": len(values), "mean_mae_bpm": float(values.mean()), "mean_mae_bootstrap_95ci": bootstrap(values, SEED, args.bootstrap_replicates), "target_workouts": len(set(target_map.values())), "target_workout_guardrail": "one fixed test target per user; original valid HR points only"}
    reference_values = {
        "hierarchical_linear": load_reference(args.reference_hierarchical, "hierarchical_linear"),
        "lightgbm": load_reference(args.reference_lightgbm, "lightgbm", "intercept_speed", "0.03"),
        "residual_tcn": load_reference(args.reference_tcn, "residual_tcn", "intercept_speed", "0.1"),
    }
    references = {"adapted_fitrec_lstm": ({row["user_key"]: float(row["mae_bpm"]) for row in metrics}, target_map)}
    references.update(reference_values)
    reference_summary = []
    paired_summary = []
    common_users = set(references["adapted_fitrec_lstm"][0])
    for mapping, _targets in references.values():
        if mapping:
            common_users &= set(mapping)
    if common_users:
        for model_name, (mapping, targets) in references.items():
            if not mapping:
                continue
            if targets and any(targets.get(user, target_map[user]) != target_map[user] for user in common_users):
                raise RuntimeError(f"Target-workout guardrail failed for reference model {model_name}")
            model_values = np.asarray([mapping[user] for user in sorted(common_users)])
            reference_summary.append({"model": model_name, "users": len(model_values), "mean_mae_bpm": float(model_values.mean()), "mean_mae_bootstrap_95ci": bootstrap(model_values, SEED + len(reference_summary), args.bootstrap_replicates)})
        adapted_values = references["adapted_fitrec_lstm"][0]
        for model_name, (mapping, _targets) in references.items():
            if model_name == "adapted_fitrec_lstm" or not mapping:
                continue
            differences = np.asarray([mapping[user] - adapted_values[user] for user in sorted(common_users)])
            paired_summary.append({"comparison": f"{model_name}_minus_adapted_fitrec_lstm", "users": len(differences), "mean_difference_bpm": float(differences.mean()), "paired_bootstrap_95ci": bootstrap(differences, SEED + 100 + len(paired_summary), args.bootstrap_replicates)})
        with (args.output_dir / "adapted_fitrec_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["model", "users", "mean_mae_bpm", "mean_mae_bootstrap_95ci_low", "mean_mae_bootstrap_95ci_high"])
            writer.writeheader()
            for row in reference_summary:
                ci = row["mean_mae_bootstrap_95ci"]
                writer.writerow({**{key: value for key, value in row.items() if key != "mean_mae_bootstrap_95ci"}, "mean_mae_bootstrap_95ci_low": ci[0], "mean_mae_bootstrap_95ci_high": ci[1]})
        with (args.output_dir / "adapted_fitrec_paired_differences.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["comparison", "users", "mean_difference_bpm", "paired_bootstrap_95ci_low", "paired_bootstrap_95ci_high"])
            writer.writeheader()
            for row in paired_summary:
                ci = row["paired_bootstrap_95ci"]
                writer.writerow({**{key: value for key, value in row.items() if key != "paired_bootstrap_95ci"}, "paired_bootstrap_95ci_low": ci[0], "paired_bootstrap_95ci_high": ci[1]})
    (args.output_dir / "adapted_fitrec_lstm_summary.json").write_text(json.dumps({"analysis": "Adapted FitRec-LSTM baseline", "stage": "User-disjoint development/validation fit followed by one locked-test evaluation", "training": training, "summary": summary, "reference_summary": reference_summary, "paired_summary": paired_summary, "common_reference_users": len(common_users), "guardrail": "This is an adapted published-method baseline, not an exact reproduction of Ni et al. (2019)."}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
