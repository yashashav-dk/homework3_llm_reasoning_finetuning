"""Chain-of-thought trace generator for Roman numeral conversion puzzles."""

from __future__ import annotations

import re

from src.solvers.numeral import (
    NumeralSolver,
    _ARABIC_TO_ROMAN,
    _ROMAN_TO_ARABIC,
    _parse_direction_and_target,
)
from src.trace_generators.base import BaseTraceGenerator, CoTTrace

# ---------------------------------------------------------------------------
# Symbol tables for step-by-step decomposition
# ---------------------------------------------------------------------------

_PLACE_TABLE: list[tuple[int, str, str]] = [
    # (place_value, name,  symbols)
    (100, "hundreds", "C / XC / L / XL"),
    (10,  "tens",     "X / IX / V / IV"),
    (1,   "ones",     "I / V"),
]

# Arabic → Roman components in descending value order
_COMPONENTS: list[tuple[int, str]] = [
    (100, "C"),
    (90,  "XC"),
    (50,  "L"),
    (40,  "XL"),
    (10,  "X"),
    (9,   "IX"),
    (5,   "V"),
    (4,   "IV"),
    (1,   "I"),
]

# Roman symbol values for parsing
_SYMBOL_VALUE: dict[str, int] = {
    "I": 1, "V": 5, "X": 10, "L": 50, "C": 100,
}


# ---------------------------------------------------------------------------
# Decomposition helpers
# ---------------------------------------------------------------------------

def _arabic_to_roman_steps(n: int) -> list[str]:
    """Return a list of prose steps for converting an Arabic integer to Roman."""
    steps: list[str] = []
    steps.append(f"I need to convert {n} to a Roman numeral.")
    steps.append(
        "Standard symbols: I=1, V=5, X=10, L=50, C=100; "
        "subtractive pairs: IV=4, IX=9, XL=40, XC=90."
    )

    remainder = n
    parts: list[str] = []
    decomp_lines: list[str] = []

    for value, symbol in _COMPONENTS:
        count = remainder // value
        if count > 0:
            contribution = symbol * count
            decomp_lines.append(
                f"  {remainder} ÷ {value} → {count} × '{symbol}' = '{contribution}'"
                f"  (remainder {remainder - value * count})"
            )
            parts.append(contribution)
            remainder -= value * count

    steps.append("Decomposing by place value (largest first):")
    steps.extend(decomp_lines)
    steps.append(
        "Concatenating all parts: " + " + ".join(f"'{p}'" for p in parts)
        + f" = '{_ARABIC_TO_ROMAN[n]}'"
    )
    return steps


def _roman_to_arabic_steps(roman: str) -> list[str]:
    """Return a list of prose steps for converting a Roman numeral to Arabic."""
    steps: list[str] = []
    steps.append(f"I need to convert the Roman numeral '{roman}' to an Arabic integer.")
    steps.append(
        "Rule: if a smaller symbol precedes a larger one, subtract it; otherwise add it."
    )

    symbol_list = list(roman.upper())
    running = 0
    calc_lines: list[str] = []

    for i, sym in enumerate(symbol_list):
        cur_val = _SYMBOL_VALUE[sym]
        if i + 1 < len(symbol_list):
            next_val = _SYMBOL_VALUE[symbol_list[i + 1]]
        else:
            next_val = 0

        if cur_val < next_val:
            calc_lines.append(
                f"  '{sym}' ({cur_val}) < next '{symbol_list[i+1]}' ({next_val})"
                f" → subtract: running = {running} - {cur_val} = {running - cur_val}"
            )
            running -= cur_val
        else:
            calc_lines.append(
                f"  '{sym}' ({cur_val}) → add: running = {running} + {cur_val}"
                f" = {running + cur_val}"
            )
            running += cur_val

    steps.append("Processing each symbol left to right:")
    steps.extend(calc_lines)
    steps.append(f"Final sum: {_ROMAN_TO_ARABIC[roman.upper()]}")
    return steps


# ---------------------------------------------------------------------------
# Trace generator
# ---------------------------------------------------------------------------


class NumeralTraceGenerator(BaseTraceGenerator):
    category = "numeral"

    def generate_trace(self, puzzle: dict, solver_result) -> CoTTrace:
        prompt: str = puzzle.get("prompt", "")
        direction, target = _parse_direction_and_target(prompt)
        answer = solver_result.predicted_answer

        lines: list[str] = []

        # Step 1: Analyze examples
        lines.append(
            "Step 1: Let me analyze the examples to understand the conversion rule."
        )
        # Pull example pairs from the prompt for illustration
        example_lines = [
            ln.strip()
            for ln in prompt.splitlines()
            if "=" in ln and ln.strip().split("=")[-1].strip()
        ]
        if example_lines:
            lines.append("Examples from the puzzle:")
            for ex in example_lines[:5]:  # cap at 5 for brevity
                lines.append(f"  {ex}")

        # Step 2: Identify direction
        lines.append("")
        lines.append("Step 2: Identify the conversion direction.")
        if direction == "arabic_to_roman":
            lines.append(
                "The examples show Arabic integers on the left and Roman numerals "
                "on the right → this is an Arabic-to-Roman conversion."
            )
            lines.append(f"Target: convert the Arabic number {target} to Roman.")
        else:
            lines.append(
                "The examples show Roman numerals on the left and Arabic integers "
                "on the right → this is a Roman-to-Arabic conversion."
            )
            lines.append(f"Target: convert the Roman numeral '{target}' to Arabic.")

        # Step 3 & 4: Perform conversion with explanation
        lines.append("")
        if direction == "arabic_to_roman":
            lines.append("Step 3: Convert Arabic → Roman numeral step by step.")
            try:
                n = int(target)
                for step in _arabic_to_roman_steps(n):
                    lines.append(step)
            except ValueError:
                lines.append(f"  (Could not parse target '{target}' as integer.)")
        else:
            lines.append("Step 3: Convert Roman → Arabic integer step by step.")
            for step in _roman_to_arabic_steps(target):
                lines.append(step)

        # Step 5: Final answer
        lines.append("")
        lines.append("Step 5: State the final answer.")
        lines.append(f"The answer is {answer}")

        thinking_text = "\n".join(lines)

        return CoTTrace(
            puzzle_id=puzzle.get("id", ""),
            category=self.category,
            thinking_text=thinking_text,
            final_answer=answer,
            token_count=self.count_tokens(thinking_text),
            is_verified=solver_result.is_correct,
        )


# ---------------------------------------------------------------------------
# CLI: generate traces to an output directory
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    import os
    import sys

    parser = argparse.ArgumentParser(
        description="Generate CoT traces for numeral puzzles."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to puzzles_classified.jsonl",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory to write trace JSONL files",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    solver = NumeralSolver()
    generator = NumeralTraceGenerator()

    out_path = os.path.join(args.output, "numeral_traces.jsonl")
    generated = skipped = 0

    with open(args.input) as fh, open(out_path, "w") as out_fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            puzzle = json.loads(line)
            if puzzle.get("category") != "numeral":
                continue
            solver_result = solver.solve(puzzle)
            if not solver_result.predicted_answer:
                skipped += 1
                print(
                    f"SKIP  id={puzzle.get('id', '?')} (no answer produced)",
                    file=sys.stderr,
                )
                continue
            trace = generator.generate_trace(puzzle, solver_result)
            record = {
                "puzzle_id": trace.puzzle_id,
                "category": trace.category,
                "thinking_text": trace.thinking_text,
                "final_answer": trace.final_answer,
                "token_count": trace.token_count,
                "is_verified": trace.is_verified,
            }
            out_fh.write(json.dumps(record) + "\n")
            generated += 1

    print(
        f"Generated {generated} traces → {out_path}"
        + (f" ({skipped} skipped)" if skipped else "")
    )
