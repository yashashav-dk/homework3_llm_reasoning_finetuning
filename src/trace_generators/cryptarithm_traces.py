"""Chain-of-thought trace generator for cryptarithm puzzles.

Only generates traces for puzzles where the solver returned confidence > 0.
"""

from __future__ import annotations

from src.solvers.base import SolverResult
from src.solvers.cryptarithm import (
    CryptarithmPuzzle,
    _all_letters,
    _assignment_valid,
    _parse_prompt,
    _try_brute_force,
    _try_carry_free,
    _try_concatenation,
    _try_reverse_concatenation,
)
from src.trace_generators.base import BaseTraceGenerator, CoTTrace


class CryptarithmTraceGenerator(BaseTraceGenerator):
    category = "cryptarithm"

    def generate_trace(self, puzzle: dict, solver_result: SolverResult) -> CoTTrace:
        puzzle_id = puzzle.get("id", "")
        prompt: str = puzzle.get("prompt", "")

        # Only generate traces when the solver found an answer
        if solver_result.confidence == 0.0 or not solver_result.predicted_answer:
            return CoTTrace(
                puzzle_id=puzzle_id,
                category=self.category,
                thinking_text="This puzzle could not be solved by the available strategies.",
                final_answer="",
                token_count=0,
                is_verified=False,
            )

        # Re-derive the puzzle structure for a self-consistent trace
        parsed = _parse_prompt(prompt)
        lines: list[str] = []

        # ------------------------------------------------------------------ #
        # Step 1 – parse the expression                                       #
        # ------------------------------------------------------------------ #
        lines.append("Let me work through this cryptarithm step by step.\n")
        lines.append("=== Step 1: Parse the arithmetic expression ===\n")

        if parsed is not None:
            lines.append(f"Expression: {parsed.raw_expr}")
            lines.append(
                f"Operands : {', '.join(parsed.operands)}"
            )
            lines.append(f"Operator : {parsed.operator}")
            lines.append(f"Result   : {parsed.result}")

            all_letters = _all_letters(parsed)
            lines.append(f"\nUnique letters involved: {', '.join(all_letters)}")
            lines.append(
                f"Each letter maps to a unique digit 0-9 "
                f"({len(all_letters)} letters total)."
            )

            if parsed.given:
                lines.append("\nPre-assigned digits:")
                for letter, digit in sorted(parsed.given.items()):
                    lines.append(f"  {letter} = {digit}")
            else:
                lines.append("\nNo digits are pre-assigned — all must be deduced.")
        else:
            lines.append("(Could not fully parse expression — using solver result directly.)")
        lines.append("")

        # ------------------------------------------------------------------ #
        # Step 2 – describe the strategy                                      #
        # ------------------------------------------------------------------ #
        lines.append("=== Step 2: Choose solving strategy ===\n")

        method = solver_result.solve_method

        if method == "given_assignment":
            lines.append(
                "The digit for the target letter is directly given in the problem "
                "statement — no deduction needed."
            )
        elif method == "concatenation":
            lines.append(
                "Strategy: CONCATENATION DETECTION\n"
                "Check whether the result word is simply the operand words joined "
                "together. If so, the digit mapping is structurally forced."
            )
        elif method == "reverse_concatenation":
            lines.append(
                "Strategy: REVERSE CONCATENATION DETECTION\n"
                "Check whether the result word equals the operands concatenated "
                "then reversed."
            )
        elif method == "carry_free_addition":
            lines.append(
                "Strategy: CARRY-FREE COLUMN ADDITION\n"
                "Work column by column right-to-left. When only one letter in a "
                "column is unknown, its digit can be deduced directly from the "
                "column sum constraint (no carry assumed)."
            )
        elif method == "brute_force":
            lines.append(
                "Strategy: BRUTE-FORCE ENUMERATION (≤4 unknown letters)\n"
                "There are few enough unknowns that we can try all valid digit "
                "permutations and check which one satisfies the equation."
            )
        else:
            lines.append(f"Strategy: {method}")
        lines.append("")

        # ------------------------------------------------------------------ #
        # Step 3 – show digit assignments                                     #
        # ------------------------------------------------------------------ #
        lines.append("=== Step 3: Digit assignments ===\n")

        if parsed is not None:
            # Re-run the matching strategy to obtain the full assignment dict
            assignment: dict[str, int] | None = None

            if method == "given_assignment":
                assignment = dict(parsed.given)

            elif method == "concatenation":
                assignment = _try_concatenation(parsed)
                if assignment is None:
                    assignment = dict(parsed.given)

            elif method == "reverse_concatenation":
                assignment = _try_reverse_concatenation(parsed)
                if assignment is None:
                    assignment = dict(parsed.given)

            elif method == "carry_free_addition":
                assignment = _try_carry_free(parsed)
                if assignment is None:
                    assignment = dict(parsed.given)

            elif method == "brute_force":
                assignment = _try_brute_force(parsed)
                if assignment is None:
                    assignment = dict(parsed.given)

            else:
                assignment = dict(parsed.given)

            if assignment:
                lines.append("Resolved letter-to-digit mapping:")
                for letter in sorted(assignment.keys()):
                    marker = " ← target" if letter == parsed.question_letter else ""
                    lines.append(f"  {letter} = {assignment[letter]}{marker}")
            else:
                lines.append("(Assignment could not be fully reconstructed for trace.)")

            # Walk through carry-free column logic explicitly when applicable
            if method == "carry_free_addition" and len(parsed.operands) >= 2:
                lines.append("\nColumn-by-column deduction (right to left):")
                max_len = len(parsed.result)
                padded = [op.rjust(max_len, " ") for op in parsed.operands]
                for col in range(max_len - 1, -1, -1):
                    col_letters = [p[col] for p in padded if p[col].strip()]
                    res_letter = parsed.result[col]
                    col_str = " + ".join(col_letters)
                    if assignment:
                        col_vals = [str(assignment.get(ch, "?")) for ch in col_letters]
                        res_val = str(assignment.get(res_letter, "?"))
                        lines.append(
                            f"  Column {max_len - col} (rightmost=1): "
                            f"{col_str} → {' + '.join(col_vals)} = {res_val}  "
                            f"({res_letter}={res_val})"
                        )
                    else:
                        lines.append(
                            f"  Column {max_len - col}: {col_str} = {res_letter}"
                        )

            # Walk through brute-force enumeration briefly
            if method == "brute_force" and parsed is not None:
                all_l = _all_letters(parsed)
                known = dict(parsed.given)
                unknowns = [ch for ch in all_l if ch not in known]
                lines.append(
                    f"\nBrute-force search: {len(unknowns)} unknown letter(s): "
                    + ", ".join(unknowns)
                )
                if assignment:
                    chosen = {ch: assignment[ch] for ch in unknowns if ch in assignment}
                    lines.append(
                        "Found valid assignment: "
                        + ", ".join(f"{ch}={d}" for ch, d in sorted(chosen.items()))
                    )
        else:
            lines.append(
                f"The solver determined: "
                f"{solver_result.solve_method} → answer = {solver_result.predicted_answer}"
            )
        lines.append("")

        # ------------------------------------------------------------------ #
        # Step 4 – verify the solution                                        #
        # ------------------------------------------------------------------ #
        lines.append("=== Step 4: Verify the solution ===\n")

        if parsed is not None and assignment:
            # Compute numeric values and check
            def word_to_num(w: str, a: dict[str, int]) -> str:
                if all(ch in a for ch in w):
                    digits = "".join(str(a[ch]) for ch in w)
                    return digits
                return "?"

            operand_strs = [word_to_num(op, assignment) for op in parsed.operands]
            result_str = word_to_num(parsed.result, assignment)

            if parsed.operator == "+":
                try:
                    op_vals = [int(s) for s in operand_strs if s != "?"]
                    res_val = int(result_str) if result_str != "?" else None
                    lhs_str = " + ".join(
                        f"{op} = {num}" for op, num in zip(parsed.operands, operand_strs)
                    )
                    lines.append(f"Check: {lhs_str}")
                    if op_vals and res_val is not None:
                        computed = sum(op_vals)
                        check = "✓ VALID" if computed == res_val else "✗ INVALID"
                        lines.append(
                            f"  {' + '.join(str(v) for v in op_vals)} = {computed}  "
                            f"and {parsed.result} = {res_val}  → {check}"
                        )
                    else:
                        lines.append("  (Partial assignment — cannot fully verify)")
                except ValueError:
                    lines.append("  (Could not evaluate expression numerically)")
            else:
                lines.append(
                    f"Operands: {', '.join(f'{op}={s}' for op, s in zip(parsed.operands, operand_strs))}"
                )
                lines.append(f"Result  : {parsed.result} = {result_str}")
        else:
            lines.append(
                f"Using solver result: answer = {solver_result.predicted_answer!r}  "
                f"(method: {solver_result.solve_method})"
            )
        lines.append("")

        # ------------------------------------------------------------------ #
        # Step 5 – answer the specific question                               #
        # ------------------------------------------------------------------ #
        lines.append("=== Step 5: Answer the question ===\n")

        if parsed is not None:
            if parsed.question_type == "letter_to_digit":
                lines.append(
                    f"Question: What digit does {parsed.question_letter} represent?"
                )
                if assignment and parsed.question_letter in assignment:
                    digit = assignment[parsed.question_letter]
                    lines.append(
                        f"From the mapping table, {parsed.question_letter} = {digit}."
                    )
                    lines.append(f"\nTherefore, {parsed.question_letter} represents digit {digit}.")
                else:
                    lines.append(
                        f"Answer: {solver_result.predicted_answer}"
                    )
            else:
                lines.append(
                    f"Question: Which letter represents digit {parsed.question_digit}?"
                )
                if assignment and parsed.question_digit is not None:
                    found = [
                        ch for ch, d in assignment.items()
                        if d == parsed.question_digit
                    ]
                    if found:
                        lines.append(
                            f"From the mapping table, digit {parsed.question_digit} "
                            f"is represented by {found[0]}."
                        )
                        lines.append(
                            f"\nTherefore, the letter representing {parsed.question_digit} "
                            f"is {found[0]}."
                        )
                    else:
                        lines.append(f"Answer: {solver_result.predicted_answer}")
                else:
                    lines.append(f"Answer: {solver_result.predicted_answer}")
        else:
            lines.append(f"Answer: {solver_result.predicted_answer}")

        lines.append("")
        lines.append(f"The answer is {solver_result.predicted_answer}")

        thinking_text = "\n".join(lines)

        return CoTTrace(
            puzzle_id=puzzle_id,
            category=self.category,
            thinking_text=thinking_text,
            final_answer=solver_result.predicted_answer,
            token_count=self.count_tokens(thinking_text),
            is_verified=solver_result.is_correct,
        )


# ---------------------------------------------------------------------------
# CLI: demo on hand-crafted cryptarithm puzzles
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.solvers.cryptarithm import CryptarithmSolver

    demo_puzzles = [
        {
            "id": "cryptarithm_demo_001",
            "category": "cryptarithm_deduce",
            "prompt": (
                "Each letter represents a unique digit 0-9:\n"
                "  AB\n"
                "+ CD\n"
                "------\n"
                " EFG\n\n"
                "Given: A=1, B=2, C=3, D=4\n"
                "What digit does E represent?"
            ),
            "answer": "0",
        },
        {
            "id": "cryptarithm_demo_002",
            "category": "cryptarithm_deduce",
            "prompt": (
                "Each letter represents a unique digit 0-9:\n"
                "  ONE\n"
                "+ TWO\n"
                "------\n"
                " THREE\n\n"
                "What digit does O represent?"
            ),
            "answer": "",  # unknown ground truth for demo
        },
    ]

    solver = CryptarithmSolver()
    generator = CryptarithmTraceGenerator()

    for demo in demo_puzzles:
        print(f"\n{'='*60}")
        print(f"Puzzle: {demo['id']}")
        print(f"Prompt:\n{demo['prompt']}")
        result = solver.solve(demo)
        print(f"\nSolver result: {result}")

        if result.confidence > 0.0:
            trace = generator.generate_trace(demo, result)
            print("\n=== CoT Trace ===")
            print(trace.thinking_text)
            print(f"\nFinal answer : {trace.final_answer!r}")
            print(f"Token count  : {trace.token_count}")
            print(f"Verified     : {trace.is_verified}")
        else:
            print("\n(Puzzle unsolvable — no trace generated)")
