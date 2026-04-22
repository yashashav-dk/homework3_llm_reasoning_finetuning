"""Chain-of-thought trace generator for gravity category puzzles.

Produces deterministic, human-readable reasoning traces that walk through:
  1. Identifying (time, distance) example pairs from the prompt
  2. Deriving the constant rate from the first example
  3. Verifying the rate against subsequent examples
  4. Applying the rate to the target time
  5. Formatting the answer to exactly two decimal places
"""

import re
import statistics

from src.solvers.base import SolverResult
from src.trace_generators.base import BaseTraceGenerator, CoTTrace


class GravityTraceGenerator(BaseTraceGenerator):
    category = "gravity"

    _EXAMPLE_PATTERN = re.compile(
        r"(?:time\s*(?:is|=)\s*|t\s*=\s*)([0-9]+(?:\.[0-9]+)?)"
        r".*?"
        r"(?:distance\s*(?:is|=)\s*|d\s*=\s*)([0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE,
    )

    _TARGET_PATTERN = re.compile(
        r"(?:what\b|find\b)"
        r".*?"
        r"(?:when\s+)?(?:time\s*(?:is|=)\s*|t\s*=\s*)([0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE,
    )

    _WHEN_PATTERN = re.compile(
        r"when\s+(?:time\s*(?:is|=)\s*|t\s*=\s*)([0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE,
    )

    def generate_trace(self, puzzle: dict, solver_result: SolverResult) -> CoTTrace:
        puzzle_id = puzzle.get("id", "")
        prompt: str = puzzle.get("prompt", "")

        # --- Re-parse examples (same logic as solver for determinism) ---
        example_pairs: list[tuple[float, float]] = []
        for m in self._EXAMPLE_PATTERN.finditer(prompt):
            example_pairs.append((float(m.group(1)), float(m.group(2))))

        rates: list[float] = [d / (t ** 2) for t, d in example_pairs if t != 0]
        rate = statistics.mean(rates) if rates else 0.0

        # --- Extract target time (mirrors GravitySolver three-strategy logic) ---
        target_match = self._TARGET_PATTERN.search(prompt)
        if target_match:
            target_time = float(target_match.group(1))
        else:
            when_match = self._WHEN_PATTERN.search(prompt)
            if when_match:
                target_time = float(when_match.group(1))
            else:
                last_end = 0
                for m in self._EXAMPLE_PATTERN.finditer(prompt):
                    last_end = m.end()
                fb = re.search(
                    r"(?:time\s*(?:is|=)\s*|t\s*=\s*)([0-9]+(?:\.[0-9]+)?)",
                    prompt[last_end:],
                    re.IGNORECASE,
                )
                target_time = float(fb.group(1)) if fb else 0.0

        answer = rate * (target_time ** 2)
        answer_str = f"{answer:.2f}"

        # --- Build the trace ---
        lines: list[str] = []

        lines.append("## Step 1 — Identify the example pairs")
        lines.append("")
        lines.append(
            "The puzzle follows the quadratic law  d = rate × t²,  "
            "where 'rate' is a constant we need to determine."
        )
        lines.append("")
        if example_pairs:
            lines.append("From the prompt I can extract the following (time, distance) pairs:")
            for i, (t, d) in enumerate(example_pairs, start=1):
                t_str = f"{t:g}"
                d_str = f"{d:.2f}"
                lines.append(f"  Example {i}:  t = {t_str},  d = {d_str}")
        else:
            lines.append("  (No example pairs found — cannot proceed.)")
        lines.append("")

        lines.append("## Step 2 — Derive the rate from the first example")
        lines.append("")
        if example_pairs:
            t0, d0 = example_pairs[0]
            t0_str = f"{t0:g}"
            d0_str = f"{d0:.2f}"
            t0_sq = t0 ** 2
            rate_from_first = d0 / t0_sq if t0 != 0 else 0.0
            lines.append(
                f"  rate = distance / time²"
            )
            lines.append(
                f"       = {d0_str} / {t0_str}²"
            )
            lines.append(
                f"       = {d0_str} / {t0_sq:g}"
            )
            lines.append(
                f"       = {rate_from_first:.4f}"
            )
        lines.append("")

        lines.append("## Step 3 — Verify rate with remaining examples")
        lines.append("")
        if len(example_pairs) >= 2:
            all_ok = True
            for i, (t, d) in enumerate(example_pairs[1:], start=2):
                t_str = f"{t:g}"
                t_sq = t ** 2
                predicted_d = rate * t_sq
                predicted_d_str = f"{predicted_d:.2f}"
                actual_d_str = f"{d:.2f}"
                match = abs(predicted_d - d) / (abs(d) + 1e-12) < 0.01
                check_mark = "✓" if match else "✗"
                lines.append(
                    f"  Check example {i}:  "
                    f"rate × t² = {rate:.4f} × {t_str}² = {rate:.4f} × {t_sq:g} "
                    f"= {predicted_d_str}  (actual: {actual_d_str}) {check_mark}"
                )
                if not match:
                    all_ok = False
            lines.append("")
            if all_ok:
                lines.append(
                    "  All examples agree — the rate is consistent at "
                    f"{rate:.4f}."
                )
            else:
                lines.append(
                    "  Warning: not all examples agree perfectly; using mean rate "
                    f"= {rate:.4f}."
                )
        else:
            lines.append("  Only one example provided; no additional verification possible.")
        lines.append("")

        lines.append("## Step 4 — Apply rate to the target time")
        lines.append("")
        target_str = f"{target_time:g}"
        target_sq = target_time ** 2
        lines.append(f"  Target time:  t = {target_str}")
        lines.append(f"  distance = rate × t²")
        lines.append(f"           = {rate:.4f} × {target_str}²")
        lines.append(f"           = {rate:.4f} × {target_sq:g}")
        lines.append(f"           = {answer:.4f}")
        lines.append("")

        lines.append("## Step 5 — Format the answer")
        lines.append("")
        lines.append(
            f"  Rounding to exactly two decimal places: {answer:.4f} → {answer_str}"
        )
        lines.append("")
        lines.append(f"The answer is {answer_str}")

        thinking_text = "\n".join(lines)

        return CoTTrace(
            puzzle_id=puzzle_id,
            category=self.category,
            thinking_text=thinking_text,
            final_answer=answer_str,
            token_count=self.count_tokens(thinking_text),
            is_verified=solver_result.is_correct,
        )


if __name__ == "__main__":
    from src.solvers.gravity import GravitySolver

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
    ]

    solver = GravitySolver()
    generator = GravityTraceGenerator()

    for puzzle in test_puzzles:
        result = solver.solve(puzzle)
        trace = generator.generate_trace(puzzle, result)
        print(f"=== Puzzle {trace.puzzle_id} ===")
        print(trace.thinking_text)
        print(f"Final answer: {trace.final_answer}")
        print(f"Verified: {trace.is_verified}  |  ~{trace.token_count} tokens")
        print()
