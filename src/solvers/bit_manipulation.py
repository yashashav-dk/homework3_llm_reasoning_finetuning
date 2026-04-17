"""Solver for bit-manipulation puzzles.

Each puzzle shows a set of 8-bit binary input→output example pairs and asks
for the output of a target input.  The transformation is a *per-bit* boolean
function: each output bit is determined solely by one (or more) bits of the
*same-position-indexed* input bits across examples.

Strategy (bit-serial):
  For each of the 8 output bit positions independently, search through the 52
  candidate gate functions (ordered simplest-first) with all valid input-bit
  combinations until one that perfectly fits all examples is found.  Then
  apply every identified gate to the target input to produce the 8-bit answer.

Gate hierarchy
--------------
Level 0 (2)  : constants — always 0 / always 1
Level 1 (2)  : identity / NOT  (single input bit)
Level 2 (10) : two-input gates — AND, OR, XOR, NAND, NOR, XNOR + 4 asymmetric
Level 3 (18) : three-input gates — MAJ, CHO, PAR3, AO, OA, AX, OX, XA, XO + negations
Level 4 (20) : four-input gates — AOA, OAO, PAR4, XX (XOR4), AXA + negations
"""

from __future__ import annotations

import itertools
import re
from typing import Callable

from src.solvers.base import BaseSolver, SolverResult

# ---------------------------------------------------------------------------
# Gate definitions
# Each gate is a callable (a: int, b: int, ...) -> int  where every arg is 0/1
# ---------------------------------------------------------------------------

# Level 0 — constants (no inputs, but accept *args for uniform calling)
_CONST_0: Callable[..., int] = lambda *_: 0
_CONST_1: Callable[..., int] = lambda *_: 1

# Level 1 — single input
_ID: Callable[[int], int] = lambda a: a
_NOT: Callable[[int], int] = lambda a: a ^ 1

# Level 2 — two inputs
_AND:      Callable[[int, int], int] = lambda a, b: a & b
_OR:       Callable[[int, int], int] = lambda a, b: a | b
_XOR:      Callable[[int, int], int] = lambda a, b: a ^ b
_NAND:     Callable[[int, int], int] = lambda a, b: (a & b) ^ 1
_NOR:      Callable[[int, int], int] = lambda a, b: (a | b) ^ 1
_XNOR:     Callable[[int, int], int] = lambda a, b: (a ^ b) ^ 1
_AND_NB:   Callable[[int, int], int] = lambda a, b: a & (b ^ 1)   # A AND NOT-B
_NA_AND:   Callable[[int, int], int] = lambda a, b: (a ^ 1) & b   # NOT-A AND B
_OR_NB:    Callable[[int, int], int] = lambda a, b: a | (b ^ 1)   # A OR NOT-B
_NA_OR:    Callable[[int, int], int] = lambda a, b: (a ^ 1) | b   # NOT-A OR B

# Level 3 — three inputs
def _maj(a: int, b: int, c: int) -> int:      return (a & b) | (b & c) | (a & c)
def _cho(a: int, b: int, c: int) -> int:      return (a & b) | ((a ^ 1) & c)       # MUX: a?b:c
def _par3(a: int, b: int, c: int) -> int:     return a ^ b ^ c
def _ao(a: int, b: int, c: int) -> int:       return a & (b | c)
def _oa(a: int, b: int, c: int) -> int:       return a | (b & c)
def _ax(a: int, b: int, c: int) -> int:       return a & (b ^ c)
def _ox(a: int, b: int, c: int) -> int:       return a | (b ^ c)
def _xa(a: int, b: int, c: int) -> int:       return a ^ (b & c)
def _xo(a: int, b: int, c: int) -> int:       return a ^ (b | c)

# Negated Level-3 variants
def _nmaj(a: int, b: int, c: int) -> int:     return _maj(a, b, c) ^ 1
def _ncho(a: int, b: int, c: int) -> int:     return _cho(a, b, c) ^ 1
def _npar3(a: int, b: int, c: int) -> int:    return _par3(a, b, c) ^ 1
def _nao(a: int, b: int, c: int) -> int:      return _ao(a, b, c) ^ 1
def _noa(a: int, b: int, c: int) -> int:      return _oa(a, b, c) ^ 1
def _nax(a: int, b: int, c: int) -> int:      return _ax(a, b, c) ^ 1
def _nox(a: int, b: int, c: int) -> int:      return _ox(a, b, c) ^ 1
def _nxa(a: int, b: int, c: int) -> int:      return _xa(a, b, c) ^ 1
def _nxo(a: int, b: int, c: int) -> int:      return _xo(a, b, c) ^ 1

# Level 4 — four inputs
def _aoa(a: int, b: int, c: int, d: int) -> int:   return (a & b) | (c & d)
def _oao(a: int, b: int, c: int, d: int) -> int:   return (a | b) & (c | d)
def _par4(a: int, b: int, c: int, d: int) -> int:  return a ^ b ^ c ^ d
def _xx(a: int, b: int, c: int, d: int) -> int:    return a ^ b ^ c ^ d           # alias PAR4
def _axa(a: int, b: int, c: int, d: int) -> int:   return (a & b) ^ (c & d)

# Negated Level-4 variants
def _naoa(a: int, b: int, c: int, d: int) -> int:  return _aoa(a, b, c, d) ^ 1
def _noao(a: int, b: int, c: int, d: int) -> int:  return _oao(a, b, c, d) ^ 1
def _npar4(a: int, b: int, c: int, d: int) -> int: return _par4(a, b, c, d) ^ 1
def _nxx(a: int, b: int, c: int, d: int) -> int:   return _xx(a, b, c, d) ^ 1
def _naxa(a: int, b: int, c: int, d: int) -> int:  return _axa(a, b, c, d) ^ 1

# Four-input gates with asymmetric structure (A op (B op C op D))
def _a_and_or3(a: int, b: int, c: int, d: int) -> int:  return a & (b | c | d)
def _a_or_and3(a: int, b: int, c: int, d: int) -> int:  return a | (b & c & d)
def _a_xor3(a: int, b: int, c: int, d: int) -> int:     return a ^ (b ^ c ^ d)
def _na_and_or3(a: int, b: int, c: int, d: int) -> int: return _a_and_or3(a, b, c, d) ^ 1
def _na_or_and3(a: int, b: int, c: int, d: int) -> int: return _a_or_and3(a, b, c, d) ^ 1


# ---------------------------------------------------------------------------
# Gate registry organised by arity
# ---------------------------------------------------------------------------

# Each entry: (name, arity, function)
GateSpec = tuple[str, int, Callable[..., int]]

GATES_LEVEL0: list[GateSpec] = [
    ("CONST_0", 0, _CONST_0),
    ("CONST_1", 0, _CONST_1),
]

GATES_LEVEL1: list[GateSpec] = [
    ("ID", 1, _ID),
    ("NOT", 1, _NOT),
]

GATES_LEVEL2: list[GateSpec] = [
    ("AND",    2, _AND),
    ("OR",     2, _OR),
    ("XOR",    2, _XOR),
    ("NAND",   2, _NAND),
    ("NOR",    2, _NOR),
    ("XNOR",   2, _XNOR),
    ("AND_NB", 2, _AND_NB),
    ("NA_AND", 2, _NA_AND),
    ("OR_NB",  2, _OR_NB),
    ("NA_OR",  2, _NA_OR),
]

GATES_LEVEL3: list[GateSpec] = [
    ("MAJ",   3, _maj),
    ("CHO",   3, _cho),
    ("PAR3",  3, _par3),
    ("AO",    3, _ao),
    ("OA",    3, _oa),
    ("AX",    3, _ax),
    ("OX",    3, _ox),
    ("XA",    3, _xa),
    ("XO",    3, _xo),
    ("NMAJ",  3, _nmaj),
    ("NCHO",  3, _ncho),
    ("NPAR3", 3, _npar3),
    ("NAO",   3, _nao),
    ("NOA",   3, _noa),
    ("NAX",   3, _nax),
    ("NOX",   3, _nox),
    ("NXA",   3, _nxa),
    ("NXO",   3, _nxo),
]

GATES_LEVEL4: list[GateSpec] = [
    ("AOA",      4, _aoa),
    ("OAO",      4, _oao),
    ("PAR4",     4, _par4),
    ("XX",       4, _xx),
    ("AXA",      4, _axa),
    ("NAOA",     4, _naoa),
    ("NOAO",     4, _noao),
    ("NPAR4",    4, _npar4),
    ("NXX",      4, _nxx),
    ("NAXA",     4, _naxa),
    ("A_AND_OR3",  4, _a_and_or3),
    ("A_OR_AND3",  4, _a_or_and3),
    ("A_XOR3",     4, _a_xor3),
    ("NA_AND_OR3", 4, _na_and_or3),
    ("NA_OR_AND3", 4, _na_or_and3),
    # Padding to reach 20 entries: additional asymmetric variants
    ("NA_XOR3",    4, lambda a, b, c, d: _a_xor3(a, b, c, d) ^ 1),
    ("AND_OAO",    4, lambda a, b, c, d: a & ((b | c) & d)),
    ("NAND_OAO",   4, lambda a, b, c, d: (a & ((b | c) & d)) ^ 1),
    ("OR_AOA",     4, lambda a, b, c, d: a | ((b & c) | d)),
    ("NOR_AOA",    4, lambda a, b, c, d: (a | ((b & c) | d)) ^ 1),
]

ALL_GATE_LEVELS: list[list[GateSpec]] = [
    GATES_LEVEL0,
    GATES_LEVEL1,
    GATES_LEVEL2,
    GATES_LEVEL3,
    GATES_LEVEL4,
]

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_BINARY_PAIR_RE = re.compile(
    r"Input\s*:\s*([01]{8})\s*[→\->=]+\s*Output\s*:\s*([01]{8})",
    re.IGNORECASE,
)
_TARGET_RE = re.compile(
    r"(?:input\s+(?:is\s+)?|target\s+(?:input\s+)?|for\s+input\s+)"
    r"([01]{8})"
    r"(?:\s*[?:]|\s*$)",
    re.IGNORECASE,
)
# Also match bare "10101010?" or "10101010" at end of a line
_BARE_BINARY_RE = re.compile(r"\b([01]{8})\b")


def _parse_examples(prompt: str) -> list[tuple[str, str]]:
    """Return list of (input_binary, output_binary) 8-bit string pairs."""
    pairs = _BINARY_PAIR_RE.findall(prompt)
    if pairs:
        return pairs

    # Fallback: look for lines with two 8-bit strings separated by an arrow
    results: list[tuple[str, str]] = []
    arrow_re = re.compile(r"([01]{8})\s*[→\->]+\s*([01]{8})")
    for m in arrow_re.finditer(prompt):
        results.append((m.group(1), m.group(2)))
    if results:
        return results

    # Last resort: collect all 8-bit strings in order, pair them up
    all_bits = _BARE_BINARY_RE.findall(prompt)
    if len(all_bits) >= 2:
        # Each consecutive non-overlapping pair is (input, output)
        results = []
        for i in range(0, len(all_bits) - 1, 2):
            results.append((all_bits[i], all_bits[i + 1]))
        return results

    return []


def _parse_target(prompt: str, examples: list[tuple[str, str]]) -> str:
    """Extract the target 8-bit input from the prompt."""
    # Try the structured target pattern
    for m in _TARGET_RE.finditer(prompt):
        candidate = m.group(1)
        # Exclude strings already seen as example inputs
        if candidate not in {inp for inp, _ in examples}:
            return candidate

    # Fallback: the last 8-bit string in the prompt that isn't in examples
    all_bits = _BARE_BINARY_RE.findall(prompt)
    example_inputs = {inp for inp, _ in examples}
    example_outputs = {out for _, out in examples}
    for candidate in reversed(all_bits):
        if candidate not in example_inputs and candidate not in example_outputs:
            return candidate

    # If still not found, take the very last 8-bit string
    if all_bits:
        return all_bits[-1]

    return "00000000"


# ---------------------------------------------------------------------------
# Core bit-serial solver logic
# ---------------------------------------------------------------------------

def _find_gate_for_bit(
    out_pos: int,
    examples: list[tuple[str, str]],
) -> tuple[str, list[int]] | None:
    """Search for the simplest gate function that produces examples[*].output[out_pos]
    from some combination of input bit positions.

    Returns (gate_name, [input_bit_positions]) or None if nothing found.
    The gate_name encodes arity; input_bit_positions has length == arity.
    """
    # Pre-extract bits for each example
    # input_bits[k] = list of 8 ints (bits 0-7 of example k's input)
    # output_bits[k] = int (the out_pos bit of example k's output)
    input_bits: list[list[int]] = [
        [int(inp[i]) for i in range(8)] for inp, _ in examples
    ]
    output_bits: list[int] = [int(out[out_pos]) for _, out in examples]

    def _check(gate_fn: Callable[..., int], positions: tuple[int, ...]) -> bool:
        for k, out_bit in enumerate(output_bits):
            args = tuple(input_bits[k][p] for p in positions)
            if gate_fn(*args) != out_bit:
                return False
        return True

    for level_gates in ALL_GATE_LEVELS:
        for gate_name, arity, gate_fn in level_gates:
            if arity == 0:
                if _check(gate_fn, ()):
                    return gate_name, []
            elif arity == 1:
                for p in range(8):
                    if _check(gate_fn, (p,)):
                        return gate_name, [p]
            else:
                # For arity >= 2 we try all ordered permutations of distinct bit
                # positions — order matters because some gates are asymmetric.
                for combo in itertools.permutations(range(8), arity):
                    if _check(gate_fn, combo):
                        return gate_name, list(combo)

    return None


def _apply_gate(
    gate_name: str,
    positions: list[int],
    input_str: str,
) -> int:
    """Apply the named gate to the specified bit positions of input_str."""
    # Resolve the function from registry
    gate_fn = _gate_fn_by_name(gate_name)
    args = tuple(int(input_str[p]) for p in positions)
    return gate_fn(*args)


def _gate_fn_by_name(name: str) -> Callable[..., int]:
    for level_gates in ALL_GATE_LEVELS:
        for gate_name, _arity, gate_fn in level_gates:
            if gate_name == name:
                return gate_fn
    raise KeyError(f"Unknown gate: {name}")


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------

class BitManipulationSolver(BaseSolver):
    """Solves bit-manipulation puzzles via exhaustive per-bit gate search."""

    category = "bit_manipulation"

    def solve(self, puzzle: dict) -> SolverResult:
        puzzle_id = puzzle.get("id", "")
        prompt: str = puzzle.get("prompt", "")
        ground_truth: str = puzzle.get("answer", "")

        # 1. Parse examples and target
        examples = _parse_examples(prompt)
        target = _parse_target(prompt, examples)

        if not examples:
            # Cannot solve without examples
            return SolverResult(
                puzzle_id=puzzle_id,
                category=self.category,
                predicted_answer="00000000",
                is_correct=False,
                solve_method="no_examples",
                confidence=0.0,
            )

        # 2. For each of the 8 output bit positions, find the gate
        bit_gates: list[tuple[str, list[int]] | None] = []
        for out_pos in range(8):
            result = _find_gate_for_bit(out_pos, examples)
            bit_gates.append(result)

        # 3. Apply gates to target input
        output_bits: list[str] = []
        confidence_sum = 0.0
        for out_pos, gate_info in enumerate(bit_gates):
            if gate_info is None:
                # Fallback: copy input bit (identity)
                output_bits.append(target[out_pos])
                confidence_sum += 0.0
            else:
                gate_name, positions = gate_info
                bit_val = _apply_gate(gate_name, positions, target)
                output_bits.append(str(bit_val))
                confidence_sum += 1.0

        predicted = "".join(output_bits)
        confidence = confidence_sum / 8.0

        is_correct = predicted.strip() == ground_truth.strip()

        return SolverResult(
            puzzle_id=puzzle_id,
            category=self.category,
            predicted_answer=predicted,
            is_correct=is_correct,
            solve_method="bit_serial_gate_search",
            confidence=confidence,
        )

    def explain(self, puzzle: dict) -> list[tuple[int, str, list[int]]]:
        """Return [(out_pos, gate_name, input_positions), ...] for all 8 bits.

        Useful for trace generation.  Positions are 0-indexed left-to-right.
        """
        prompt: str = puzzle.get("prompt", "")
        examples = _parse_examples(prompt)
        result = []
        for out_pos in range(8):
            gate_info = _find_gate_for_bit(out_pos, examples)
            if gate_info is None:
                result.append((out_pos, "ID", [out_pos]))
            else:
                gate_name, positions = gate_info
                result.append((out_pos, gate_name, positions))
        return result


# ---------------------------------------------------------------------------
# CLI: accuracy evaluation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    import sys

    # Quick demo to verify the solver works on a NOT puzzle
    _demo = {
        "id": "bit_demo_001",
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
        "answer": "00110101",
    }
    _solver = BitManipulationSolver()
    _res = _solver.solve(_demo)
    print(f"Demo: predicted={_res.predicted_answer!r} expected={_demo['answer']!r} correct={_res.is_correct}")
    print()

    from src.metrics.competition import verify as competition_verify

    parser = argparse.ArgumentParser(
        description="Evaluate BitManipulationSolver on classified puzzles."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to puzzles_classified.jsonl",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print wrong predictions.",
    )
    args = parser.parse_args()

    solver = BitManipulationSolver()
    total = correct = 0

    with open(args.input) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            puzzle = json.loads(line)
            if puzzle.get("category") != "bit_manipulation":
                continue
            result = solver.solve(puzzle)
            total += 1
            if competition_verify(puzzle["answer"], result.predicted_answer):
                correct += 1
            elif args.verbose:
                print(
                    f"WRONG  id={puzzle.get('id', '?')} "
                    f"predicted={result.predicted_answer!r} "
                    f"expected={puzzle['answer']!r}",
                    file=sys.stderr,
                )

    if total == 0:
        print("No bit_manipulation puzzles found.")
    else:
        accuracy = correct / total
        print(f"Bit manipulation accuracy: {correct}/{total} = {accuracy:.4f}")
