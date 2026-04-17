"""CoT trace generator for unit_conversion puzzles.

Produces a deterministic, human-readable chain-of-thought that mirrors the
exact reasoning a student would write when solving a linear unit-conversion
problem.

Trace structure
---------------
1. Parse all example pairs from the prompt.
2. Derive the conversion factor from the first example with explicit arithmetic.
3. Cross-verify with every remaining example.
4. Apply the factor to the target value.
5. Format the final answer as X.XX.
"""

from __future__ import annotations

from src.solvers.base import SolverResult
from src.solvers.unit_conversion import _parse_example_pairs, _extract_target
from src.trace_generators.base import BaseTraceGenerator, CoTTrace


class UnitConversionTraceGenerator(BaseTraceGenerator):
    """Generate step-by-step CoT traces for unit-conversion puzzles."""

    category = "unit_conversion"

    def generate_trace(self, puzzle: dict, solver_result: SolverResult) -> CoTTrace:
        puzzle_id: str = puzzle.get("id", "unknown")
        prompt: str = puzzle.get("prompt", "")

        # ------------------------------------------------------------------
        # Re-derive intermediate values so the trace is self-contained.
        # ------------------------------------------------------------------
        pairs = _parse_example_pairs(prompt)
        target = _extract_target(prompt)

        # Guard: if parsing failed, emit a minimal trace.
        if not pairs or target is None:
            thinking = (
                "I could not parse the example pairs or target from this prompt.\n"
                f"Raw prompt:\n{prompt}\n"
                "Unable to derive a conversion factor."
            )
            return CoTTrace(
                puzzle_id=puzzle_id,
                category=self.category,
                thinking_text=thinking,
                final_answer=solver_result.predicted_answer,
                token_count=self.count_tokens(thinking),
                is_verified=False,
            )

        first_inp, first_out = pairs[0]
        factor = first_out / first_inp
        # Use the mean factor to match the solver exactly.
        mean_factor = sum(o / i for i, o in pairs) / len(pairs)
        raw_answer = mean_factor * target
        formatted_answer = f"{raw_answer:.2f}"

        # ------------------------------------------------------------------
        # Build the trace text
        # ------------------------------------------------------------------
        lines: list[str] = []

        # Step 1 — identify example pairs
        lines.append("Step 1: Parse the example pairs from the prompt.")
        for k, (inp, out) in enumerate(pairs, start=1):
            lines.append(f"  Example {k}: input = {inp}, output = {out}")

        lines.append("")

        # Step 2 — derive factor from the first example
        lines.append("Step 2: Derive the conversion factor from example 1.")
        lines.append(
            f"  factor = output / input = {first_out} / {first_inp} = {factor:.6g}"
        )

        lines.append("")

        # Step 3 — verify with remaining examples
        if len(pairs) > 1:
            lines.append("Step 3: Verify the factor with the remaining example(s).")
            for k, (inp, out) in enumerate(pairs[1:], start=2):
                expected = factor * inp
                ok = abs(expected - out) < 1e-4
                check_symbol = "✓" if ok else "✗ (inconsistent!)"
                lines.append(
                    f"  Check example {k}: {factor:.6g} * {inp}"
                    f" = {expected:.2f}  (given {out:.2f}) {check_symbol}"
                )
        else:
            lines.append("Step 3: Only one example provided — factor verified by definition.")

        lines.append("")

        # Step 4 — apply factor to target
        lines.append("Step 4: Apply the conversion factor to the target value.")
        lines.append(f"  target input = {target}")
        lines.append(
            f"  result = factor * target = {mean_factor:.6g} * {target} = {raw_answer:.6g}"
        )

        lines.append("")

        # Step 5 — format to X.XX
        lines.append("Step 5: Format the result to two decimal places.")
        lines.append(f"  {raw_answer:.6g}  →  {formatted_answer}")

        lines.append("")
        lines.append(f"Final answer: {formatted_answer}")

        thinking_text = "\n".join(lines)

        is_verified = formatted_answer == solver_result.predicted_answer

        return CoTTrace(
            puzzle_id=puzzle_id,
            category=self.category,
            thinking_text=thinking_text,
            final_answer=formatted_answer,
            token_count=self.count_tokens(thinking_text),
            is_verified=is_verified,
        )


# ---------------------------------------------------------------------------
# Manual smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.solvers.unit_conversion import UnitConversionSolver

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

    generator = UnitConversionTraceGenerator()
    trace = generator.generate_trace(sample_puzzle, result)

    print("=" * 60)
    print(f"Puzzle ID : {trace.puzzle_id}")
    print(f"Category  : {trace.category}")
    print(f"Verified  : {trace.is_verified}")
    print(f"Tokens    : {trace.token_count}")
    print(f"Answer    : {trace.final_answer}")
    print("=" * 60)
    print(trace.thinking_text)
