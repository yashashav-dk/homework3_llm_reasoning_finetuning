"""Solver for Roman numeral conversion puzzles (Arabic↔Roman, 1–100)."""

from __future__ import annotations

import re

from src.solvers.base import BaseSolver, SolverResult

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

_ARABIC_TO_ROMAN: dict[int, str] = {}
_ROMAN_TO_ARABIC: dict[str, int] = {}


def _build_lookup() -> None:
    """Populate both lookup dicts for 1–100 using standard subtractive rules."""
    _parts = [
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    for n in range(1, 101):
        remainder = n
        roman = ""
        for value, symbol in _parts:
            while remainder >= value:
                roman += symbol
                remainder -= value
        _ARABIC_TO_ROMAN[n] = roman
        _ROMAN_TO_ARABIC[roman] = n


_build_lookup()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ROMAN_RE = re.compile(r"[IVXLC]+")
_ARABIC_RE = re.compile(r"\b([1-9][0-9]?|100)\b")


def _is_roman(token: str) -> bool:
    return bool(re.fullmatch(r"[IVXLC]+", token))


def _is_arabic(token: str) -> bool:
    try:
        v = int(token)
        return 1 <= v <= 100
    except ValueError:
        return False


def _parse_direction_and_target(prompt: str) -> tuple[str, str]:
    """Return (direction, target_token).

    direction is one of:
      'arabic_to_roman' – input is an Arabic integer, output is a Roman numeral
      'roman_to_arabic' – input is a Roman numeral, output is an Arabic integer

    Strategy:
    1. Collect all example pairs (input → output) from lines containing '='.
    2. Infer direction from the first example.
    3. Find the target: the last token on the final question line (before '=').
    """
    lines = prompt.strip().splitlines()

    # Collect example pairs: lines of the form  "<lhs> = <rhs>"
    examples: list[tuple[str, str]] = []
    question_lhs: str | None = None

    for line in lines:
        line = line.strip()
        if "=" not in line:
            continue
        left, _, right = line.partition("=")
        lhs = left.strip()
        rhs = right.strip()
        if rhs:
            examples.append((lhs, rhs))
        else:
            # This is the question: "Convert X =" or "X ="
            question_lhs = lhs

    # Infer direction from examples
    direction = "arabic_to_roman"  # default
    if examples:
        sample_lhs, sample_rhs = examples[0]
        # Extract last whitespace-delimited token from lhs
        lhs_token = sample_lhs.split()[-1] if sample_lhs.split() else sample_lhs
        if _is_arabic(lhs_token):
            direction = "arabic_to_roman"
        elif _is_roman(lhs_token.upper()):
            direction = "roman_to_arabic"
        else:
            # Check rhs to flip
            rhs_token = sample_rhs.split()[-1] if sample_rhs.split() else sample_rhs
            if _is_arabic(rhs_token):
                direction = "roman_to_arabic"

    # Find the target from the question line or last example line
    target = ""
    if question_lhs is not None:
        tokens = question_lhs.split()
        # Walk tokens from the end to find the meaningful value
        for tok in reversed(tokens):
            tok_clean = tok.strip(",:;")
            if direction == "arabic_to_roman" and _is_arabic(tok_clean):
                target = tok_clean
                break
            elif direction == "roman_to_arabic" and _is_roman(tok_clean.upper()):
                target = tok_clean.upper()
                break
        if not target and tokens:
            # Fallback: last token
            target = tokens[-1].strip(",:;")
    else:
        # No explicit question line; try to infer from last line of prompt
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            for tok in reversed(tokens):
                tok_clean = tok.strip("=,:;?")
                if direction == "arabic_to_roman" and _is_arabic(tok_clean):
                    target = tok_clean
                    break
                elif direction == "roman_to_arabic" and _is_roman(tok_clean.upper()):
                    target = tok_clean.upper()
                    break
            if target:
                break

    return direction, target


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class NumeralSolver(BaseSolver):
    category = "numeral"

    def solve(self, puzzle: dict) -> SolverResult:
        puzzle_id = puzzle.get("id", "")
        prompt: str = puzzle.get("prompt", "")
        ground_truth: str = puzzle.get("answer", "")

        direction, target = _parse_direction_and_target(prompt)

        predicted = ""
        solve_method = direction

        if direction == "arabic_to_roman":
            try:
                n = int(target)
                predicted = _ARABIC_TO_ROMAN.get(n, "")
            except (ValueError, KeyError):
                predicted = ""
        else:  # roman_to_arabic
            target_upper = target.upper()
            predicted = str(_ROMAN_TO_ARABIC.get(target_upper, ""))

        is_correct = predicted.lower() == ground_truth.strip().lower()

        return SolverResult(
            puzzle_id=puzzle_id,
            category=self.category,
            predicted_answer=predicted,
            is_correct=is_correct,
            solve_method=solve_method,
            confidence=1.0 if predicted else 0.0,
        )


# ---------------------------------------------------------------------------
# CLI: accuracy evaluation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    import sys

    from src.metrics.competition import verify

    parser = argparse.ArgumentParser(
        description="Evaluate NumeralSolver on classified puzzles."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to puzzles_classified.jsonl",
    )
    args = parser.parse_args()

    solver = NumeralSolver()
    total = correct = 0

    with open(args.input) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            puzzle = json.loads(line)
            if puzzle.get("category") != "numeral":
                continue
            result = solver.solve(puzzle)
            total += 1
            if verify(puzzle["answer"], result.predicted_answer):
                correct += 1
            else:
                print(
                    f"WRONG  id={puzzle.get('id', '?')} "
                    f"predicted={result.predicted_answer!r} "
                    f"expected={puzzle['answer']!r}",
                    file=sys.stderr,
                )

    if total == 0:
        print("No numeral puzzles found.")
    else:
        accuracy = correct / total
        print(f"Numeral accuracy: {correct}/{total} = {accuracy:.4f}")
