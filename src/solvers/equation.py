"""Solver for equation (transformation rules) category puzzles.

Each puzzle shows examples: A<sym>B = result, where <sym> is a secret operator
symbol. The same symbol always uses the same transformation rule within a puzzle.
Different symbols may use different rules.

The solver discovers each symbol's rule via brute-force over:
- Operand transforms: identity, reverse digits
- Math operations: +, -, *, //, %, concat, etc.
- Output transforms: identity, reverse digits
"""

from __future__ import annotations

import re
from math import gcd

from src.solvers.base import BaseSolver, SolverResult


def _rev(n: int) -> int:
    """Reverse digits, preserving sign."""
    if n < 0:
        return -int(str(-n)[::-1])
    return int(str(n)[::-1])


def _safe_div(a: int, b: int) -> int | None:
    if b == 0:
        return None
    if a % b != 0:
        return None
    return a // b


# Math operations: (name, func(a, b) -> int|None)
_OPERATIONS: list[tuple[str, object]] = [
    ("a+b", lambda a, b: a + b),
    ("a-b", lambda a, b: a - b),
    ("b-a", lambda a, b: b - a),
    ("a*b", lambda a, b: a * b),
    ("a//b", lambda a, b: _safe_div(a, b)),
    ("b//a", lambda a, b: _safe_div(b, a)),
    ("a%b", lambda a, b: a % b if b != 0 else None),
    ("b%a", lambda a, b: b % a if a != 0 else None),
    ("|a-b|", lambda a, b: abs(a - b)),
    ("max", lambda a, b: max(a, b)),
    ("min", lambda a, b: min(a, b)),
    ("gcd", lambda a, b: gcd(a, b) if a and b else None),
    ("a&b", lambda a, b: a & b),
    ("a|b", lambda a, b: a | b),
    ("a^b", lambda a, b: a ^ b),
    ("concat_ab", lambda a, b: int(str(a) + str(b))),
    ("concat_ba", lambda a, b: int(str(b) + str(a))),
    ("a**b", lambda a, b: a ** b if 0 <= b <= 10 and a < 1000 else None),
    ("b**a", lambda a, b: b ** a if 0 <= a <= 10 and b < 1000 else None),
    ("dsum_a+dsum_b", lambda a, b: sum(int(d) for d in str(abs(a))) + sum(int(d) for d in str(abs(b)))),
    ("dsum_a*dsum_b", lambda a, b: sum(int(d) for d in str(abs(a))) * sum(int(d) for d in str(abs(b)))),
    ("a*b+a+b", lambda a, b: a * b + a + b),
    ("a*b-1", lambda a, b: a * b - 1),
    ("a*b+1", lambda a, b: a * b + 1),
    ("a*b-a", lambda a, b: a * b - a),
    ("a*b+a", lambda a, b: a * b + a),
    ("a*b-b", lambda a, b: a * b - b),
    ("a*b+b", lambda a, b: a * b + b),
    ("a*a+b", lambda a, b: a * a + b),
    ("a+b*b", lambda a, b: a + b * b),
    ("a*a-b", lambda a, b: a * a - b),
    ("a-b*b", lambda a, b: a - b * b),
    ("a*a*b", lambda a, b: a * a * b),
    ("a*b*b", lambda a, b: a * b * b),
    ("(a+b)*2", lambda a, b: (a + b) * 2),
    ("(a-b)*2", lambda a, b: (a - b) * 2),
    ("(a+b)**2", lambda a, b: (a + b) ** 2),
    ("(a-b)**2", lambda a, b: (a - b) ** 2),
    ("a*10+b", lambda a, b: a * 10 + b),
    ("b*10+a", lambda a, b: b * 10 + a),
]

# Operand transforms
_OPERAND_TRANSFORMS = [
    ("id", lambda x: x),
    ("rev", _rev),
]

# Output transforms
_OUTPUT_TRANSFORMS = [
    ("id", lambda x: x),
    ("rev", _rev),
]


def _result_matches(predicted_int: int, expected_str: str) -> bool:
    """Check if a predicted integer matches the expected result string.

    Handles leading zeros (e.g., predicted=75 vs expected='0075') and negatives.
    """
    predicted_str = str(predicted_int)
    # Handle negative results
    if expected_str.startswith('-') and predicted_str.startswith('-'):
        return _result_matches(-predicted_int, expected_str[1:])
    if expected_str.startswith('-') != predicted_str.startswith('-'):
        return False
    expected_clean = expected_str.lstrip('0') or '0'
    predicted_clean = predicted_str.lstrip('0') or '0'
    return predicted_clean == expected_clean


def _parse_examples(prompt: str) -> tuple[list[tuple[int, str, int, str]], tuple[int, str, int] | None]:
    """Parse prompt into (examples, target).

    examples: list of (a, symbol, b, result_str)
    target: (a, symbol, b)
    """
    examples = []
    target = None

    for line in prompt.split('\n'):
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if 'wonderland' in low or 'transformation' in low or 'below' in low:
            continue

        # Target: "Now, determine the result for: 75*97"
        m_target = re.search(r'result\s+for:\s*(\d+)\s*([^\d\s]+)\s*(\d+)', line)
        if m_target:
            target = (int(m_target.group(1)), m_target.group(2), int(m_target.group(3)))
            continue

        if 'now,' in low or 'determine' in low:
            continue

        # Example: "52{43 = 9"
        m = re.match(r'(\d+)\s*([^\d\s=]+)\s*(\d+)\s*=\s*(.+)', line)
        if m:
            a = int(m.group(1))
            sym = m.group(2)
            b = int(m.group(3))
            result_str = m.group(4).strip()
            examples.append((a, sym, b, result_str))

    return examples, target


def _find_rule(examples: list[tuple[int, int, str]]):
    """Find the rule that explains all (a, b, result_str) triples.

    Returns (op_func, ot_func, out_func, description) or None.
    """
    for ot_name, ot_func in _OPERAND_TRANSFORMS:
        for op_name, op_func in _OPERATIONS:
            for out_name, out_func in _OUTPUT_TRANSFORMS:
                all_match = True
                for a, b, result_str in examples:
                    try:
                        raw = op_func(ot_func(a), ot_func(b))
                        if raw is None:
                            all_match = False
                            break
                        predicted = out_func(raw)
                        if not _result_matches(predicted, result_str):
                            all_match = False
                            break
                    except (ValueError, OverflowError, ZeroDivisionError):
                        all_match = False
                        break
                if all_match:
                    desc = f"{ot_name}_{op_name}_{out_name}"
                    return op_func, ot_func, out_func, desc
    return None


class EquationSolver(BaseSolver):
    category = "equation"

    def solve(self, puzzle: dict) -> SolverResult:
        puzzle_id = puzzle.get("id", "")
        prompt = puzzle.get("prompt", "")
        ground_truth = puzzle.get("answer", "")

        examples, target = _parse_examples(prompt)

        if not target:
            return SolverResult(
                puzzle_id=puzzle_id, category=self.category,
                predicted_answer="", is_correct=False,
                solve_method="no_target", confidence=0.0,
            )

        target_a, target_sym, target_b = target

        # Group examples by symbol
        sym_examples: dict[str, list[tuple[int, int, str]]] = {}
        for a, sym, b, result_str in examples:
            sym_examples.setdefault(sym, []).append((a, b, result_str))

        # Strategy 1: Find rule for target symbol specifically
        target_sym_examples = sym_examples.get(target_sym, [])
        rule = _find_rule(target_sym_examples) if target_sym_examples else None

        # Strategy 2: If target symbol not in examples, or rule not found,
        # try treating all examples as one rule
        if rule is None:
            all_examples = [(a, b, r) for a, _, b, r in examples]
            rule = _find_rule(all_examples)

        if rule is None:
            return SolverResult(
                puzzle_id=puzzle_id, category=self.category,
                predicted_answer="", is_correct=False,
                solve_method="no_rule_found", confidence=0.0,
            )

        op_func, ot_func, out_func, desc = rule

        try:
            raw = op_func(ot_func(target_a), ot_func(target_b))
            predicted = str(out_func(raw)) if raw is not None else ""
        except (ValueError, OverflowError, ZeroDivisionError):
            predicted = ""

        from src.metrics.competition import verify
        is_correct = verify(ground_truth, predicted) if ground_truth and predicted else False

        return SolverResult(
            puzzle_id=puzzle_id, category=self.category,
            predicted_answer=predicted, is_correct=is_correct,
            solve_method=desc, confidence=1.0 if predicted else 0.0,
        )


if __name__ == "__main__":
    import argparse
    import json

    from src.metrics.competition import verify

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    solver = EquationSolver()
    total = correct = 0
    cats: dict[str, list[int]] = {}

    with open(args.input) as f:
        for line in f:
            p = json.loads(line)
            if not p["category"].startswith("equation"):
                continue
            total += 1
            cat = p["category"]
            cats.setdefault(cat, [0, 0])
            cats[cat][1] += 1
            result = solver.solve(p)
            if verify(p["answer"], result.predicted_answer):
                correct += 1
                cats[cat][0] += 1
            elif args.verbose:
                print(f"WRONG id={p['id']} expected={p['answer']!r} predicted={result.predicted_answer!r} method={result.solve_method}")

    print(f"Equation accuracy: {correct}/{total} = {correct/total:.4f}")
    for cat, (c, t) in sorted(cats.items()):
        print(f"  {cat}: {c}/{t} = {c/t:.4f}")
