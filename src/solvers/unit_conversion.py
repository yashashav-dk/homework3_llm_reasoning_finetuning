"""Solver for unit_conversion puzzles.

These puzzles show linear scaling examples (input → output pairs), derive the
conversion factor, and ask for the converted value of a target input.

Example prompt fragment::

    3 glorps is 9.00 bleeps
    5 glorps is 15.00 bleeps
    How many bleeps is 7 glorps?

Strategy:
    factor = output / input  (verified consistent across all examples)
    answer = factor * target
    formatted as "X.XX"
"""

from __future__ import annotations

import re
from typing import List, Tuple

from src.solvers.base import BaseSolver, SolverResult


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# Matches any line that contains exactly two floating-point / integer numbers.
# We treat the first as the "input" value and the second as the "output" value.
_TWO_NUMBER_RE = re.compile(
    r"""
    (?<!\d)                    # not preceded by a digit
    (\d+(?:\.\d+)?)            # group 1 — first number
    (?:\s+\S+){0,5}            # 0-5 non-numeric tokens in between
    \s+(?:is|=|to|gives?|->)?\s*  # optional connector
    (\d+(?:\.\d+)?)            # group 2 — second number
    """,
    re.VERBOSE,
)

# Simpler fallback: grab all numbers from the line, take first two.
_ALL_NUMBERS_RE = re.compile(r"\d+(?:\.\d+)?")

# Pattern to find the target in the question line.
# "How many bleeps is 7 glorps?" → target = 7
_TARGET_RE = re.compile(
    r"(?:how\s+many\s+\S+\s+(?:is|are|equals?)\s+|convert\s+|what\s+is\s+)"
    r"(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _parse_example_pairs(prompt: str) -> List[Tuple[float, float]]:
    """Return a list of (input_value, output_value) numeric pairs from the prompt.

    Each non-question line that contains exactly (or at least) two numbers is
    treated as an example.  The question line is identified as containing a
    question mark or the words "how many / what is".
    """
    pairs: List[Tuple[float, float]] = []

    for line in prompt.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Skip question lines.
        if "?" in stripped or re.search(r"\b(how\s+many|what\s+is)\b", stripped, re.I):
            continue

        numbers = _ALL_NUMBERS_RE.findall(stripped)
        if len(numbers) >= 2:
            inp = float(numbers[0])
            out = float(numbers[1])
            if inp != 0:
                pairs.append((inp, out))

    return pairs


def _extract_target(prompt: str) -> float | None:
    """Extract the target input value from the question line."""
    for line in prompt.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "?" not in stripped and not re.search(
            r"\b(how\s+many|what\s+is|convert)\b", stripped, re.I
        ):
            continue

        # Try the structured pattern first.
        m = _TARGET_RE.search(stripped)
        if m:
            return float(m.group(1))

        # Fallback: last number on the question line.
        numbers = _ALL_NUMBERS_RE.findall(stripped)
        if numbers:
            return float(numbers[-1])

    return None


class UnitConversionSolver(BaseSolver):
    """Deterministic solver for linear unit-conversion puzzles."""

    category = "unit_conversion"

    # Tolerance for factor consistency check across examples.
    _FACTOR_TOL = 1e-6

    def solve(self, puzzle: dict) -> SolverResult:
        puzzle_id: str = puzzle.get("id", "unknown")
        prompt: str = puzzle.get("prompt", "")
        ground_truth: str = puzzle.get("answer", "")

        # ------------------------------------------------------------------
        # 1. Parse example pairs
        # ------------------------------------------------------------------
        pairs = _parse_example_pairs(prompt)

        if not pairs:
            return SolverResult(
                puzzle_id=puzzle_id,
                category=self.category,
                predicted_answer="",
                is_correct=False,
                solve_method="no_examples_found",
                confidence=0.0,
            )

        # ------------------------------------------------------------------
        # 2. Derive conversion factor from the first example
        # ------------------------------------------------------------------
        factors = [out / inp for inp, out in pairs]
        primary_factor = factors[0]

        # ------------------------------------------------------------------
        # 3. Verify factor consistency
        # ------------------------------------------------------------------
        consistent = all(
            abs(f - primary_factor) <= self._FACTOR_TOL * max(1.0, abs(primary_factor))
            for f in factors
        )
        confidence = 1.0 if consistent else 0.8

        # Use the mean factor when values are slightly inconsistent (floating-point noise).
        factor = sum(factors) / len(factors)

        # ------------------------------------------------------------------
        # 4. Extract the target input value
        # ------------------------------------------------------------------
        target = _extract_target(prompt)

        if target is None:
            return SolverResult(
                puzzle_id=puzzle_id,
                category=self.category,
                predicted_answer="",
                is_correct=False,
                solve_method="target_not_found",
                confidence=0.0,
            )

        # ------------------------------------------------------------------
        # 5. Compute and format answer
        # ------------------------------------------------------------------
        raw_answer = factor * target
        predicted = f"{raw_answer:.2f}"

        is_correct = predicted == ground_truth.strip()

        return SolverResult(
            puzzle_id=puzzle_id,
            category=self.category,
            predicted_answer=predicted,
            is_correct=is_correct,
            solve_method="linear_factor",
            confidence=confidence,
        )


# ---------------------------------------------------------------------------
# Manual smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_puzzle = {
        "id": "uc_demo_001",
        "prompt": (
            "3 glorps is 9.00 bleeps\n"
            "5 glorps is 15.00 bleeps\n"
            "How many bleeps is 7 glorps?"
        ),
        "answer": "21.00",
    }

    solver = UnitConversionSolver()
    result = solver.solve(sample_puzzle)
    print(f"Puzzle   : {sample_puzzle['prompt']}")
    print(f"Predicted: {result.predicted_answer}")
    print(f"Expected : {sample_puzzle['answer']}")
    print(f"Correct  : {result.is_correct}")
    print(f"Method   : {result.solve_method}")
    print(f"Confidence: {result.confidence}")

    # Verify helper
    print(f"verify() : {solver.verify(sample_puzzle)}")
