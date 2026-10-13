#!/usr/bin/env python3
"""Validate the minimal public release and its user-level result invariants."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXPECTED_CELLS = {
    ("hierarchical_linear", "base"),
    ("hierarchical_linear", "extended"),
    ("lightgbm", "base"),
    ("lightgbm", "extended"),
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    feature = rows(RESULTS / "feature_matched_ablation" / "feature_matched_ablation_user_metrics.csv")
    fitrec = rows(RESULTS / "adapted_fitrec_lstm" / "adapted_fitrec_lstm_user_metrics.csv")
    split = rows(RESULTS / "adapted_fitrec_lstm" / "adapted_fitrec_user_split.csv")
    hierarchical = {
        row["user_key"]: row
        for row in rows(RESULTS / "reference" / "hierarchical_k5_user_metrics.csv")
    }
    sequence_reference = rows(RESULTS / "reference" / "sequence_models_k5_user_metrics.csv")
    lightgbm = {row["user_key"]: row for row in sequence_reference if row["model"] == "lightgbm"}

    by_user: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in feature:
        by_user[row["user_key"]].append(row)
    training = {row["user_key"] for row in split if row["split"] == "training"}
    validation = {row["user_key"] for row in split if row["split"] == "validation"}

    checks = {
        "feature_rows": len(feature) == 428,
        "feature_users": len(by_user) == 107,
        "feature_four_cells": all(
            {(row["model"], row["feature_set"]) for row in user_rows} == EXPECTED_CELLS
            for user_rows in by_user.values()
        ),
        "feature_one_target": all(
            len({row["target_workout_key"] for row in user_rows}) == 1
            for user_rows in by_user.values()
        ),
        "feature_finite": all(math.isfinite(float(row["mae_bpm"])) for row in feature),
        "feature_target_matches_reference": all(
            row["target_workout_key"] == hierarchical[row["user_key"]]["target_workout_key"]
            for row in feature
        ),
        "feature_points_match_reference": all(
            row["valid_points"] == lightgbm[row["user_key"]]["valid_points"]
            for row in feature
        ),
        "fitrec_users": len(fitrec) == 107,
        "fitrec_finite": all(math.isfinite(float(row["mae_bpm"])) for row in fitrec),
        "fitrec_target_matches_reference": all(
            row["target_workout_key"] == hierarchical[row["user_key"]]["target_workout_key"]
            for row in fitrec
        ),
        "fitrec_points_match_reference": all(
            row["valid_points"] == lightgbm[row["user_key"]]["valid_points"]
            for row in fitrec
        ),
        "training_users": len(training) == 470,
        "validation_users": len(validation) == 118,
        "train_validation_disjoint": not training & validation,
    }
    print(json.dumps(checks, indent=2))
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("Release validation failed: " + ", ".join(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
