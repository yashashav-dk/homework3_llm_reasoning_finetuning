"""Chain-of-thought trace generator for numeric equation puzzles.

Generates a step-by-step reasoning trace that:
  1. Parses examples and identifies input→output pairs.
  2. Announces the search for the hidden operation.
  3. Shows the discovered operation.
  4. Verifies on each known example.
  5. Applies the operation to the target inputs.
"""

from __future__ import annotations

from src.solvers.base import SolverResult
from src.solvers.equation import (
    FoundRule,
    Number,
    find_rule,
    parse_examples_and_target,
)
from src.trace_generators.base import BaseTraceGenerator, CoTTrace


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(n: Number) -> str:
    """Format a number: drop .0 suffix when the value is an integer."""
    if isinstance(n, float) and n == int(n):
        return str(int(n))
    return str(n)


def _fmt_inputs(inputs: list[Number]) -> str:
    return ", ".join(_fmt(x) for x in inputs)


def _fmt_call(inputs: list[Number], rule: FoundRule) -> str:
    """Build a human-readable expression string for the rule applied to inputs."""
    t_ins = rule.in_transform_fn(inputs)
    op = rule.op_name

    # Replace 'a' / 'b' tokens in the op_name with actual values.
    # We substitute whole-word occurrences to avoid partial replacements.
    import re
    expr = op
    if len(t_ins) >= 1:
        expr = re.sub(r"\ba\b", _fmt(t_ins[0]), expr)
    if len(t_ins) >= 2:
        expr = re.sub(r"\bb\b", _fmt(t_ins[1]), expr)

    # Apply output transform annotation if any.
    if rule.out_transform_name:
        expr = f"swap_result_digits({expr})"

    return expr


def _compute_str(inputs: list[Number], rule: FoundRule) -> tuple[str, str]:
    """Return (expression_string, result_string) for the given inputs."""
    expr = _fmt_call(inputs, rule)
    try:
        result_num = rule.apply(inputs)
        if isinstance(result_num, float) and result_num == int(result_num):
            result_num = int(result_num)
        return expr, _fmt(result_num)
    except Exception as exc:
        return expr, f"ERROR({exc})"


# ---------------------------------------------------------------------------
# Trace generator
# ---------------------------------------------------------------------------


class EquationTraceGenerator(BaseTraceGenerator):
    category = "equation"

    def generate_trace(self, puzzle: dict, solver_result: SolverResult) -> CoTTrace:
        puzzle_id = puzzle.get("id", "")
        prompt: str = puzzle.get("prompt", "")

        # ------------------------------------------------------------------ #
        # Re-derive all intermediate state.                                   #
        # ------------------------------------------------------------------ #
        examples, target_inputs = parse_examples_and_target(prompt)
        rule: FoundRule | None = find_rule(examples)

        lines: list[str] = []

        # ------------------------------------------------------------------ #
        # Step 1 – Parse examples                                             #
        # ------------------------------------------------------------------ #
        lines.append("=== Step 1: Parse examples ===\n")

        if examples:
            for i, ex in enumerate(examples, start=1):
                lines.append(
                    f"  Example {i}: f({_fmt_inputs(ex.inputs)}) = {_fmt(ex.output)}"
                )
        else:
            lines.append("  (No examples found in prompt.)")

        if target_inputs is not None:
            lines.append(f"\n  Target query: f({_fmt_inputs(target_inputs)}) = ?")
        else:
            lines.append("\n  (No target query found in prompt.)")

        lines.append("")

        # ------------------------------------------------------------------ #
        # Step 2 – Search for the operation                                   #
        # ------------------------------------------------------------------ #
        lines.append("=== Step 2: Search for the operation ===\n")
        lines.append(
            "  Scanning transforms × operators to find a rule consistent with "
            "all examples..."
        )

        if not examples:
            lines.append("  Cannot search: no examples available.\n")
        else:
            arity = len(examples[0].inputs)
            lines.append(
                f"  Puzzle arity: {arity}-input function "
                f"({'binary' if arity >= 2 else 'unary'})."
            )
            lines.append(
                "  Checking up to 4 input transforms × 32+ operators "
                "× 2 output transforms = ~256 candidates."
            )
            lines.append(
                "  Verification: every candidate must match ALL "
                f"{len(examples)} example(s) before being accepted.\n"
            )

        # ------------------------------------------------------------------ #
        # Step 3 – Report the found operation (or failure)                    #
        # ------------------------------------------------------------------ #
        lines.append("=== Step 3: Found operation ===\n")

        if rule is None:
            lines.append("  No matching operation found in the candidate set.")
            lines.append(
                "  (The puzzle may require an operation outside the search space, "
                "or the examples may be inconsistent.)\n"
            )
        else:
            # Build a readable summary of the rule.
            transform_note = (
                f"  Input transform : {rule.transform_name}"
                if rule.transform_name not in ("AB_CD", "")
                else "  Input transform : identity (use inputs as-is)"
            )
            out_note = (
                f"  Output transform: swap result digits ({rule.out_transform_name})"
                if rule.out_transform_name
                else "  Output transform: identity (use raw result)"
            )
            lines.append(f"  Found: f(a, b) = {rule.op_name}")
            lines.append(transform_note)
            lines.append(out_note)
            lines.append("")

        # ------------------------------------------------------------------ #
        # Step 4 – Verify on each example                                     #
        # ------------------------------------------------------------------ #
        lines.append("=== Step 4: Verify on all examples ===\n")

        if rule is None:
            lines.append("  (Skipped — no rule found.)\n")
        elif not examples:
            lines.append("  (No examples to verify.)\n")
        else:
            all_pass = True
            for ex in examples:
                expr, result_str = _compute_str(ex.inputs, rule)
                expected = _fmt(ex.output)
                passed = result_str == expected
                mark = "✓" if passed else "✗"
                lines.append(
                    f"  f({_fmt_inputs(ex.inputs)}) = {expr} = {result_str} "
                    f"(expected {expected}) {mark}"
                )
                if not passed:
                    all_pass = False
            lines.append("")
            if all_pass:
                lines.append(f"  All {len(examples)} example(s) verified. Rule confirmed.\n")
            else:
                lines.append(
                    "  WARNING: some examples did not verify — "
                    "rule may be incorrect.\n"
                )

        # ------------------------------------------------------------------ #
        # Step 5 – Apply to target                                            #
        # ------------------------------------------------------------------ #
        lines.append("=== Step 5: Apply to target ===\n")

        final_answer = solver_result.predicted_answer

        if target_inputs is None:
            lines.append("  (No target query — nothing to compute.)\n")
        elif rule is None:
            lines.append(
                f"  Cannot compute f({_fmt_inputs(target_inputs)}) "
                f"— no rule was found."
            )
            if final_answer:
                lines.append(f"  Fallback answer: {final_answer}\n")
            lines.append("")
        else:
            expr, result_str = _compute_str(target_inputs, rule)
            lines.append(
                f"  f({_fmt_inputs(target_inputs)}) = {expr} = {result_str}"
            )
            lines.append("")

        # ------------------------------------------------------------------ #
        # Final boxed answer                                                  #
        # ------------------------------------------------------------------ #
        lines.append(f"\\boxed{{{final_answer}}}")

        thinking_text = "\n".join(lines)

        return CoTTrace(
            puzzle_id=puzzle_id,
            category=self.category,
            thinking_text=thinking_text,
            final_answer=final_answer,
            token_count=self.count_tokens(thinking_text),
            is_verified=solver_result.is_correct,
        )


# ---------------------------------------------------------------------------
# CLI: demo on hand-crafted equation puzzles
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.solvers.equation import EquationSolver

    DEMO_PUZZLES = [
        {
            "id": "eq_demo_add_001",
            "category": "equation_numeric_deduce",
            "prompt": (
                "f(12, 34) = 46\n"
                "f(56, 78) = 134\n"
                "f(5, 3) = ?"
            ),
            "answer": "8",
        },
        {
            "id": "eq_demo_mul_001",
            "category": "equation_numeric_deduce",
            "prompt": (
                "f(3, 4) = 12\n"
                "f(7, 5) = 35\n"
                "f(6, 9) = ?"
            ),
            "answer": "54",
        },
        {
            "id": "eq_demo_rev_digits_001",
            "category": "equation_numeric_deduce",
            "prompt": (
                "f(12, 34) = 21\n"   # reverse_digits(12) = 21
                "f(45, 67) = 54\n"
                "f(89, 10) = ?"
            ),
            "answer": "98",
        },
        {
            "id": "eq_demo_unary_square_001",
            "category": "equation_numeric_deduce",
            "prompt": (
                "f(3) = 9\n"
                "f(5) = 25\n"
                "f(7) = ?"
            ),
            "answer": "49",
        },
    ]

    solver = EquationSolver()
    generator = EquationTraceGenerator()

    for demo in DEMO_PUZZLES:
        print("=" * 60)
        print(f"Puzzle: {demo['id']}")
        print(f"Prompt:\n{demo['prompt']}\n")
        result = solver.solve(demo)
        print(f"Solver result : {result}")
        trace = generator.generate_trace(demo, result)
        print("\n--- CoT Trace ---")
        print(trace.thinking_text)
        print(f"\nFinal answer : {trace.final_answer!r}")
        print(f"Token count  : {trace.token_count}")
        print(f"Verified     : {trace.is_verified}")
        print()
