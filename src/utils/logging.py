"""Append-only experiment CSV logging utility."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENT_LOG_FIELDS = [
    "run_id",
    "config_hash",
    "timestamp",
    "hypothesis",
    "treatment_variable",
    "control_run_id",
    "seed",
    "categories_included",
    "overall_accuracy",
    "per_category_accuracy",
    "decision",
]

DEFAULT_LOG_PATH = Path("experiments/experiment_log.csv")


def ensure_log_exists(path: Path = DEFAULT_LOG_PATH) -> None:
    """Create experiment log CSV with header if it doesn't exist."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(EXPERIMENT_LOG_FIELDS)


def log_experiment(
    run_id: str,
    config_hash: str,
    hypothesis: str,
    treatment_variable: str,
    seed: int,
    categories_included: list[str],
    overall_accuracy: float,
    per_category_accuracy: dict[str, float],
    decision: str,
    control_run_id: str = "",
    path: Path = DEFAULT_LOG_PATH,
) -> None:
    """Append a single experiment result to the CSV log."""
    ensure_log_exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            run_id,
            config_hash,
            datetime.now(timezone.utc).isoformat(),
            hypothesis,
            treatment_variable,
            control_run_id,
            seed,
            json.dumps(categories_included),
            f"{overall_accuracy:.6f}",
            json.dumps(per_category_accuracy),
            decision,
        ])
