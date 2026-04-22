"""Chain-of-thought trace generator for bit-manipulation puzzles.

The generated trace reflects *bit-serial* reasoning:
  1. Announce the plan.
  2. For each output bit position, state which gate was identified and why.
  3. Apply every gate to the target input one bit at a time, writing out the
     arithmetic explicitly.
  4. Concatenate all output bits and box the final answer.

This mirrors the model behaviour that achieves ~85% accuracy on the
Kaggle competition — parallel multi-bit phrasing drops accuracy to ~9%.
"""

from __future__ import annotations

from src.solvers.base import SolverResult
from src.solvers.bit_manipulation import (
    ALL_GATE_LEVELS,
    BitManipulationSolver,
    _apply_gate,
    _find_gate_for_bit,
    _gate_fn_by_name,
    _parse_examples,
    _parse_target,
)
from src.trace_generators.base import BaseTraceGenerator, CoTTrace


# ---------------------------------------------------------------------------
# Human-readable gate descriptions
# ---------------------------------------------------------------------------

_GATE_TEMPLATES: dict[str, str] = {
    # Level 0
    "CONST_0":   "constant 0 (always outputs 0 regardless of input)",
    "CONST_1":   "constant 1 (always outputs 1 regardless of input)",
    # Level 1
    "ID":        "identity — copy input bit {0}",
    "NOT":       "NOT — flip input bit {0}",
    # Level 2
    "AND":       "AND(bit_{0}, bit_{1})",
    "OR":        "OR(bit_{0}, bit_{1})",
    "XOR":       "XOR(bit_{0}, bit_{1})",
    "NAND":      "NAND(bit_{0}, bit_{1})  [NOT(bit_{0} AND bit_{1})]",
    "NOR":       "NOR(bit_{0}, bit_{1})   [NOT(bit_{0} OR bit_{1})]",
    "XNOR":      "XNOR(bit_{0}, bit_{1})  [NOT(bit_{0} XOR bit_{1})]",
    "AND_NB":    "AND(bit_{0}, NOT bit_{1})",
    "NA_AND":    "AND(NOT bit_{0}, bit_{1})",
    "OR_NB":     "OR(bit_{0}, NOT bit_{1})",
    "NA_OR":     "OR(NOT bit_{0}, bit_{1})",
    # Level 3
    "MAJ":       "MAJ(bit_{0}, bit_{1}, bit_{2})  [majority vote]",
    "CHO":       "CHO(bit_{0}, bit_{1}, bit_{2})  [bit_{0}? bit_{1}: bit_{2}]",
    "PAR3":      "PAR3(bit_{0}, bit_{1}, bit_{2})  [bit_{0} XOR bit_{1} XOR bit_{2}]",
    "AO":        "AO(bit_{0}, bit_{1}, bit_{2})   [bit_{0} AND (bit_{1} OR bit_{2})]",
    "OA":        "OA(bit_{0}, bit_{1}, bit_{2})   [bit_{0} OR (bit_{1} AND bit_{2})]",
    "AX":        "AX(bit_{0}, bit_{1}, bit_{2})   [bit_{0} AND (bit_{1} XOR bit_{2})]",
    "OX":        "OX(bit_{0}, bit_{1}, bit_{2})   [bit_{0} OR (bit_{1} XOR bit_{2})]",
    "XA":        "XA(bit_{0}, bit_{1}, bit_{2})   [bit_{0} XOR (bit_{1} AND bit_{2})]",
    "XO":        "XO(bit_{0}, bit_{1}, bit_{2})   [bit_{0} XOR (bit_{1} OR bit_{2})]",
    "NMAJ":      "NOT MAJ(bit_{0}, bit_{1}, bit_{2})",
    "NCHO":      "NOT CHO(bit_{0}, bit_{1}, bit_{2})",
    "NPAR3":     "NOT PAR3(bit_{0}, bit_{1}, bit_{2})",
    "NAO":       "NOT AO(bit_{0}, bit_{1}, bit_{2})",
    "NOA":       "NOT OA(bit_{0}, bit_{1}, bit_{2})",
    "NAX":       "NOT AX(bit_{0}, bit_{1}, bit_{2})",
    "NOX":       "NOT OX(bit_{0}, bit_{1}, bit_{2})",
    "NXA":       "NOT XA(bit_{0}, bit_{1}, bit_{2})",
    "NXO":       "NOT XO(bit_{0}, bit_{1}, bit_{2})",
    # Level 4
    "AOA":       "AOA(bit_{0}, bit_{1}, bit_{2}, bit_{3})  [(bit_{0} AND bit_{1}) OR (bit_{2} AND bit_{3})]",
    "OAO":       "OAO(bit_{0}, bit_{1}, bit_{2}, bit_{3})  [(bit_{0} OR bit_{1}) AND (bit_{2} OR bit_{3})]",
    "PAR4":      "PAR4(bit_{0}, bit_{1}, bit_{2}, bit_{3})  [XOR of all 4 bits]",
    "XX":        "XX(bit_{0}, bit_{1}, bit_{2}, bit_{3})   [XOR of all 4 bits]",
    "AXA":       "AXA(bit_{0}, bit_{1}, bit_{2}, bit_{3})  [(bit_{0} AND bit_{1}) XOR (bit_{2} AND bit_{3})]",
    "NAOA":      "NOT AOA(bit_{0}, bit_{1}, bit_{2}, bit_{3})",
    "NOAO":      "NOT OAO(bit_{0}, bit_{1}, bit_{2}, bit_{3})",
    "NPAR4":     "NOT PAR4(bit_{0}, bit_{1}, bit_{2}, bit_{3})",
    "NXX":       "NOT XX(bit_{0}, bit_{1}, bit_{2}, bit_{3})",
    "NAXA":      "NOT AXA(bit_{0}, bit_{1}, bit_{2}, bit_{3})",
    "A_AND_OR3": "bit_{0} AND (bit_{1} OR bit_{2} OR bit_{3})",
    "A_OR_AND3": "bit_{0} OR (bit_{1} AND bit_{2} AND bit_{3})",
    "A_XOR3":    "bit_{0} XOR (bit_{1} XOR bit_{2} XOR bit_{3})",
    "NA_AND_OR3":"NOT [bit_{0} AND (bit_{1} OR bit_{2} OR bit_{3})]",
    "NA_OR_AND3":"NOT [bit_{0} OR (bit_{1} AND bit_{2} AND bit_{3})]",
    "NA_XOR3":   "NOT [bit_{0} XOR (bit_{1} XOR bit_{2} XOR bit_{3})]",
    "AND_OAO":   "bit_{0} AND ((bit_{1} OR bit_{2}) AND bit_{3})",
    "NAND_OAO":  "NOT [bit_{0} AND ((bit_{1} OR bit_{2}) AND bit_{3})]",
    "OR_AOA":    "bit_{0} OR ((bit_{1} AND bit_{2}) OR bit_{3})",
    "NOR_AOA":   "NOT [bit_{0} OR ((bit_{1} AND bit_{2}) OR bit_{3})]",
}


def _gate_description(gate_name: str, positions: list[int]) -> str:
    """Format a human-readable gate description with concrete bit indices."""
    template = _GATE_TEMPLATES.get(gate_name, gate_name)
    return template.format(*positions)


def _gate_level(gate_name: str) -> int:
    """Return the level (0-4) of a gate."""
    for level, level_gates in enumerate(ALL_GATE_LEVELS):
        for name, _arity, _fn in level_gates:
            if name == gate_name:
                return level
    return -1


def _compute_step(gate_name: str, positions: list[int], input_str: str) -> str:
    """Return a one-line string showing the gate evaluation step."""
    fn = _gate_fn_by_name(gate_name)
    args = [int(input_str[p]) for p in positions]
    result = fn(*args)

    if gate_name in ("CONST_0", "CONST_1"):
        return f"constant → {result}"

    if gate_name == "ID":
        return f"input[{positions[0]}] = {args[0]} → {result}"

    if gate_name == "NOT":
        return f"NOT input[{positions[0]}] = NOT {args[0]} = {result}"

    # Two-input gates: pretty-print the Boolean expression
    _OP_SYMBOLS: dict[str, str] = {
        "AND":    "AND",
        "OR":     "OR",
        "XOR":    "XOR",
        "NAND":   "NAND",
        "NOR":    "NOR",
        "XNOR":   "XNOR",
        "AND_NB": "AND NOT",
        "NA_AND": "NOT AND",
        "OR_NB":  "OR NOT",
        "NA_OR":  "NOT OR",
    }
    if gate_name in _OP_SYMBOLS:
        op = _OP_SYMBOLS[gate_name]
        a, b = args[0], args[1]
        pa, pb = positions[0], positions[1]
        # Clarify negation variants
        if gate_name == "AND_NB":
            expr = f"input[{pa}] AND (NOT input[{pb}]) = {a} AND (NOT {b}) = {a} AND {b^1} = {result}"
        elif gate_name == "NA_AND":
            expr = f"(NOT input[{pa}]) AND input[{pb}] = (NOT {a}) AND {b} = {a^1} AND {b} = {result}"
        elif gate_name == "OR_NB":
            expr = f"input[{pa}] OR (NOT input[{pb}]) = {a} OR (NOT {b}) = {a} OR {b^1} = {result}"
        elif gate_name == "NA_OR":
            expr = f"(NOT input[{pa}]) OR input[{pb}] = (NOT {a}) OR {b} = {a^1} OR {b} = {result}"
        else:
            expr = f"input[{pa}] {op} input[{pb}] = {a} {op} {b} = {result}"
        return expr

    # Fallback for higher-arity gates: show argument list
    args_str = ", ".join(
        f"input[{p}]={v}" for p, v in zip(positions, args)
    )
    return f"{gate_name}({args_str}) = {result}"


# ---------------------------------------------------------------------------
# Trace generator
# ---------------------------------------------------------------------------

class BitManipulationTraceGenerator(BaseTraceGenerator):
    """Generate bit-serial CoT traces for bit-manipulation puzzles."""

    category = "bit_manipulation"

    def generate_trace(self, puzzle: dict, solver_result: SolverResult) -> CoTTrace:
        puzzle_id = puzzle.get("id", "")
        prompt: str = puzzle.get("prompt", "")

        # Re-derive all state (deterministic, independent of solver_result internals)
        examples = _parse_examples(prompt)
        target = _parse_target(prompt, examples)

        # Identify the gate for each output bit position
        bit_gates: list[tuple[str, list[int]]] = []
        for out_pos in range(8):
            info = _find_gate_for_bit(out_pos, examples)
            if info is None:
                bit_gates.append(("ID", [out_pos]))  # fallback: identity
            else:
                bit_gates.append(info)

        lines: list[str] = []

        # ------------------------------------------------------------------ #
        # Step 1 — announce the plan                                          #
        # ------------------------------------------------------------------ #
        lines.append(
            "Let me analyze the examples to find the boolean function for each output bit.\n"
        )
        lines.append(
            "I will process each bit position independently (bit-serial analysis).\n"
        )

        # ------------------------------------------------------------------ #
        # Step 2 — show example data                                          #
        # ------------------------------------------------------------------ #
        lines.append("=== Example pairs ===\n")
        for i, (inp, out) in enumerate(examples, start=1):
            lines.append(f"  Example {i}: {inp} → {out}")
        lines.append("")

        # ------------------------------------------------------------------ #
        # Step 3 — identify gate for each bit position                        #
        # ------------------------------------------------------------------ #
        lines.append("=== Identifying boolean function for each output bit ===\n")

        for out_pos, (gate_name, positions) in enumerate(bit_gates):
            level = _gate_level(gate_name)
            desc = _gate_description(gate_name, positions)
            lines.append(f"Bit {out_pos}:")

            # Show the observation that led to this gate
            observed_vals: list[str] = []
            for k, (inp, out) in enumerate(examples):
                in_vals = [inp[p] for p in positions] if positions else []
                out_bit = out[out_pos]
                if not in_vals:
                    observed_vals.append(f"    Example {k+1}: output[{out_pos}] = {out_bit}")
                else:
                    args_str = ", ".join(
                        f"input[{p}]={inp[p]}" for p in positions
                    )
                    observed_vals.append(
                        f"    Example {k+1}: {args_str} → output[{out_pos}] = {out_bit}"
                    )
            lines.extend(observed_vals)

            lines.append(
                f"  → Level-{level} gate identified: {desc}"
            )
            lines.append("")

        # ------------------------------------------------------------------ #
        # Step 4 — apply gates to target (bit-serial computation)             #
        # ------------------------------------------------------------------ #
        lines.append(f"=== Applying gates to target input: {target} ===\n")

        output_bits: list[str] = []
        for out_pos, (gate_name, positions) in enumerate(bit_gates):
            step = _compute_step(gate_name, positions, target)
            result_bit = _apply_gate(gate_name, positions, target)
            output_bits.append(str(result_bit))
            lines.append(f"  Bit {out_pos}: {step}  →  output[{out_pos}] = {result_bit}")

        # ------------------------------------------------------------------ #
        # Step 5 — assemble final answer                                      #
        # ------------------------------------------------------------------ #
        predicted = "".join(output_bits)

        lines.append("")
        lines.append("=== Assembling output bits ===\n")
        bit_listing = "  " + "   ".join(
            f"[{i}]={b}" for i, b in enumerate(output_bits)
        )
        lines.append(bit_listing)
        lines.append(f"\n  Concatenated: {''.join(output_bits)}")
        lines.append("")
        lines.append(f"The answer is {predicted}")

        thinking_text = "\n".join(lines)

        return CoTTrace(
            puzzle_id=puzzle_id,
            category=self.category,
            thinking_text=thinking_text,
            final_answer=predicted,
            token_count=self.count_tokens(thinking_text),
            is_verified=solver_result.is_correct,
        )


# ---------------------------------------------------------------------------
# CLI: demo on a hand-crafted bit-manipulation puzzle
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Transformation:
    #   output bit 0 = NOT input bit 0
    #   output bits 1-6 = XOR(bit_i, bit_{i+1})   (adjacent XOR)
    #   output bit 7 = input bit 7
    demo_puzzle = {
        "id": "bitman_demo_001",
        "category": "bit_manipulation",
        "prompt": (
            "Input: 01001010 -> Output: 10110101\n"
            "Input: 11001100 -> Output: 00110011\n"
            "Input: 00001111 -> Output: 11110000\n"
            "Input: 10101010 -> Output: 01010101\n"
            "Input: 11111111 -> Output: 00000000\n"
            "Input: 00000000 -> Output: 11111111\n"
            "What is the output for input 11001010?"
        ),
        # NOT of each bit (bitwise complement)
        "answer": "00110101",
    }

    solver = BitManipulationSolver()
    result = solver.solve(demo_puzzle)
    print("Solver result:", result)
    print()

    generator = BitManipulationTraceGenerator()
    trace = generator.generate_trace(demo_puzzle, result)
    print("=== CoT Trace ===")
    print(trace.thinking_text)
    print()
    print(f"Final answer : {trace.final_answer!r}")
    print(f"Token count  : {trace.token_count}")
    print(f"Verified     : {trace.is_verified}")
