"""Solver for gravity category puzzles.

Gravity puzzles provide (time, distance) example pairs following d = rate * t^2,
analogous to d = 0.5 * g * t^2 in free-fall physics. Given the examples, the
solver derives the constant rate then applies it to a target time.
"""

import re
import statistics

from src.solvers.base import BaseSolver, SolverResult


class GravitySolver(BaseSolver):
    category = "gravity"

    # Regex to extract a (time, distance) example pair from a single sentence.
    # Handles phrasings like:
    #   "If time is 2, distance is 8.00"
    #   "t = 2, d = 8.00"
    #   "time=2 distance=8"
    _EXAMPLE_PATTERN = re.compile(
        r"(?:time\s*(?:is|=)\s*|t\s*=\s*)([0-9]+(?:\.[0-9]+)?)"
        r".*?"
        r"(?:distance\s*(?:is|=)\s*|d\s*=\s*)([0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE,
    )

    # Regex for the question sentence — must contain an interrogative marker
    # ("what", "find", "when") so it cannot fire on example sentences.
    # Handles phrasings like:
    #   "What is the distance when time is 5?"
    #   "find the distance when t = 5"
    #   "when time is 5, what is the distance?"
    _TARGET_PATTERN = re.compile(
        r"(?:what\b|find\b)"
        r".*?"
        r"(?:when\s+)?(?:time\s*(?:is|=)\s*|t\s*=\s*)([0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE,
    )

    # Looser fallback: "when time is X" anywhere in a question-like tail
    _WHEN_PATTERN = re.compile(
        r"when\s+(?:time\s*(?:is|=)\s*|t\s*=\s*)([0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE,
    )

    def solve(self, puzzle: dict) -> SolverResult:
        puzzle_id = puzzle.get("id", "")
        prompt: str = puzzle.get("prompt", "")
        ground_truth: str = puzzle.get("answer", "")

        # --- Parse example pairs ---
        example_pairs: list[tuple[float, float]] = []
        for m in self._EXAMPLE_PATTERN.finditer(prompt):
            t = float(m.group(1))
            d = float(m.group(2))
            example_pairs.append((t, d))

        if not example_pairs:
            return SolverResult(
                puzzle_id=puzzle_id,
                category=self.category,
                predicted_answer="ERROR",
                is_correct=False,
                solve_method="gravity_solver",
                confidence=0.0,
            )

        # --- Derive rate from each example ---
        rates: list[float] = []
        for t, d in example_pairs:
            if t == 0:
                continue
            rates.append(d / (t ** 2))

        if not rates:
            return SolverResult(
                puzzle_id=puzzle_id,
                category=self.category,
                predicted_answer="ERROR",
                is_correct=False,
                solve_method="gravity_solver",
                confidence=0.0,
            )

        rate = statistics.mean(rates)

        # Assess consistency: all derived rates should agree within 1 %
        consistent = all(abs(r - rate) / (abs(rate) + 1e-12) < 0.01 for r in rates)
        confidence = 1.0 if consistent else 0.8

        # --- Extract target time ---
        # Strategy 1: question sentence beginning with "what" or "find"
        target_match = self._TARGET_PATTERN.search(prompt)
        if target_match is not None:
            target_time = float(target_match.group(1))
        else:
            # Strategy 2: "when time is X" / "when t = X" anywhere
            when_match = self._WHEN_PATTERN.search(prompt)
            if when_match is not None:
                target_time = float(when_match.group(1))
            else:
                # Strategy 3: any "time is X" / "t = X" that appears AFTER
                # the last example pair — i.e. in the question tail
                last_example_end = 0
                for m in self._EXAMPLE_PATTERN.finditer(prompt):
                    last_example_end = m.end()
                tail = prompt[last_example_end:]
                tail_match = re.search(
                    r"(?:time\s*(?:is|=)\s*|t\s*=\s*)([0-9]+(?:\.[0-9]+)?)",
                    tail,
                    re.IGNORECASE,
                )
                if tail_match is None:
                    return SolverResult(
                        puzzle_id=puzzle_id,
                        category=self.category,
                        predicted_answer="ERROR",
                        is_correct=False,
                        solve_method="gravity_solver",
                        confidence=0.0,
                    )
                target_time = float(tail_match.group(1))

        # --- Compute answer ---
        answer = rate * (target_time ** 2)
        predicted_answer = f"{answer:.2f}"

        # --- Check correctness ---
        is_correct = False
        if ground_truth:
            try:
                is_correct = abs(float(ground_truth) - answer) / (abs(float(ground_truth)) + 1e-12) < 0.01
            except ValueError:
                is_correct = predicted_answer == ground_truth.strip()

        return SolverResult(
            puzzle_id=puzzle_id,
            category=self.category,
            predicted_answer=predicted_answer,
            is_correct=is_correct,
            solve_method="gravity_solver",
            confidence=confidence,
        )


if __name__ == "__main__":
    # Verification examples
    test_puzzles = [
        {
            "id": "g001",
            "prompt": (
                "If time is 2, distance is 8.00. "
                "If time is 3, distance is 18.00. "
                "What is the distance when time is 5?"
            ),
            "answer": "50.00",
        },
        {
            "id": "g002",
            "prompt": (
                "t = 1, d = 4.90. "
                "t = 4, d = 78.40. "
                "What is the distance when t = 6?"
            ),
            "answer": "176.40",
        },
        {
            "id": "g003",
            "prompt": (
                "If time is 3, distance is 44.10. "
                "If time is 5, distance is 122.50. "
                "Find the distance when time is 7."
            ),
            "answer": "240.10",
        },
    ]

    solver = GravitySolver()
    for puzzle in test_puzzles:
        result = solver.solve(puzzle)
        status = "PASS" if result.is_correct else "FAIL"
        print(
            f"[{status}] id={result.puzzle_id}  "
            f"predicted={result.predicted_answer}  "
            f"expected={puzzle['answer']}  "
            f"confidence={result.confidence}"
        )
