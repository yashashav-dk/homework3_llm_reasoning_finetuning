"""
Create a stratified train/validation split from a list of puzzle dicts.

Usage:
    python splits.py --input data/puzzles_classified.jsonl \
                     --output data/splits/ \
                     --val-pct 0.10 \
                     --seed 42
"""

import argparse
import json
import os
import random
from collections import defaultdict


def create_split(
    puzzles: list[dict],
    val_pct: float = 0.10,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Return (train_ids, val_ids) as a stratified split by category.

    Each category contributes proportionally to the validation set.
    Within each category the selection is deterministic (given *seed*).

    Parameters
    ----------
    puzzles:
        List of dicts, each containing at least "id" and "category" keys.
    val_pct:
        Fraction of puzzles to allocate to validation (per category).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    (train_ids, val_ids)
        Two lists of puzzle id strings.
    """
    if not 0.0 < val_pct < 1.0:
        raise ValueError(f"val_pct must be in (0, 1), got {val_pct}")

    rng = random.Random(seed)

    # Group puzzle ids by category.
    by_category: dict[str, list[str]] = defaultdict(list)
    for puzzle in puzzles:
        by_category[puzzle["category"]].append(puzzle["id"])

    train_ids: list[str] = []
    val_ids: list[str] = []

    for category in sorted(by_category):  # sorted for determinism
        ids = by_category[category]
        # Shuffle within category using the shared rng.
        shuffled = ids[:]
        rng.shuffle(shuffled)

        # At least 1 sample goes to val when the category has >= 2 puzzles,
        # otherwise everything goes to train.
        n_val = max(1, round(len(shuffled) * val_pct)) if len(shuffled) >= 2 else 0
        val_ids.extend(shuffled[:n_val])
        train_ids.extend(shuffled[n_val:])

    return train_ids, val_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a stratified train/val split from puzzles_classified.jsonl."
    )
    parser.add_argument(
        "--input",
        default="data/puzzles_classified.jsonl",
        help="Path to puzzles_classified.jsonl (default: data/puzzles_classified.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="data/splits/",
        help="Output directory for train_ids.json and val_ids.json (default: data/splits/)",
    )
    parser.add_argument(
        "--val-pct",
        type=float,
        default=0.10,
        help="Validation fraction (default: 0.10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    # Load puzzles.
    puzzles: list[dict] = []
    with open(args.input, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                puzzles.append(json.loads(line))

    train_ids, val_ids = create_split(puzzles, val_pct=args.val_pct, seed=args.seed)

    # Write outputs.
    os.makedirs(args.output, exist_ok=True)
    train_path = os.path.join(args.output, "train_ids.json")
    val_path = os.path.join(args.output, "val_ids.json")

    with open(train_path, "w", encoding="utf-8") as fh:
        json.dump(train_ids, fh, indent=2)

    with open(val_path, "w", encoding="utf-8") as fh:
        json.dump(val_ids, fh, indent=2)

    # Print statistics per category.
    id_to_category: dict[str, str] = {p["id"]: p["category"] for p in puzzles}
    train_set = set(train_ids)
    val_set = set(val_ids)

    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0})
    for pid in train_ids:
        by_category[id_to_category[pid]]["train"] += 1
    for pid in val_ids:
        by_category[id_to_category[pid]]["val"] += 1

    print(f"{'Category':<30} {'Train':>7} {'Val':>7} {'Total':>7} {'Val%':>7}")
    print("-" * 60)
    for category in sorted(by_category):
        counts = by_category[category]
        total = counts["train"] + counts["val"]
        pct = 100.0 * counts["val"] / total if total else 0.0
        print(f"{category:<30} {counts['train']:>7} {counts['val']:>7} {total:>7} {pct:>6.1f}%")
    print("-" * 60)
    total_train = len(train_ids)
    total_val = len(val_ids)
    total_all = total_train + total_val
    overall_pct = 100.0 * total_val / total_all if total_all else 0.0
    print(f"{'TOTAL':<30} {total_train:>7} {total_val:>7} {total_all:>7} {overall_pct:>6.1f}%")
    print(f"\nWrote {train_path}")
    print(f"Wrote {val_path}")

    # Sanity checks.
    assert len(train_set) + len(val_set) == len(puzzles), "IDs lost or duplicated"
    assert train_set.isdisjoint(val_set), "Train and val sets overlap"


if __name__ == "__main__":
    main()
