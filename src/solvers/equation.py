"""Solver for numeric equation puzzles (equation_numeric_deduce / equation_numeric_guess).

Strategy (winning Kaggle approach):
  - 4 operand transforms × 32+ operators = ~130 candidate rules tested in
    frequency order.
  - EX2 verification: a candidate must pass ALL parsed examples, not just the
    first one, before being accepted.
  - If no rule is found (unsolvable or ambiguous), the solver falls back to
    returning the most common output seen in the examples.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable

from src.solvers.base import BaseSolver, SolverResult

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Number = int | float


@dataclass
class Example:
    inputs: list[Number]
    output: Number


@dataclass
class Rule:
    transform_name: str
    op_name: str
    op_fn: Callable[..., Number]
    arity: int  # 1 or 2


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _to_num(s: str) -> Number:
    f = float(s)
    return int(f) if f == int(f) else f


def _parse_examples(prompt: str) -> tuple[list[Example], list[Number] | None]:
    """Return (examples_with_known_output, target_inputs).

    Lines of the form:
        f(12, 34) = 46
        f(56) = 7
        f(5, 3) = ?
    are parsed.  The last line whose right-hand side is '?' supplies the
    target inputs.  All other '= <number>' lines supply examples.
    """
    examples: list[Example] = []
    target_inputs: list[Number] | None = None

    for line in prompt.splitlines():
        line = line.strip()
        if not line:
            continue

        # Match:  f(...) = something
        # Also handles bare "12, 34 → 46" style lines
        m = re.match(
            r"(?:[a-zA-Z_]\w*\s*\()?"      # optional "f("
            r"([\d\s,.\-]+)"                # capture: one or more numbers (the inputs)
            r"(?:\))?"                      # optional ")"
            r"\s*[=→>:]\s*"                 # separator
            r"(\?|(?:-?\d+(?:\.\d+)?))",    # rhs: "?" or a number
            line,
        )
        if not m:
            continue

        lhs, rhs = m.group(1), m.group(2)
        nums = [_to_num(n) for n in _NUM_RE.findall(lhs)]
        if not nums:
            continue

        if rhs.strip() == "?":
            target_inputs = nums
        else:
            examples.append(Example(inputs=nums, output=_to_num(rhs)))

    return examples, target_inputs


# ---------------------------------------------------------------------------
# Digit-manipulation helpers
# ---------------------------------------------------------------------------

def _reverse_digits(n: Number) -> Number:
    """Reverse the decimal digits of |n|, preserving sign."""
    neg = n < 0
    s = str(abs(int(n)))
    rev = int(s[::-1])
    return -rev if neg else rev


def _digit_sum(n: Number) -> Number:
    return sum(int(d) for d in str(abs(int(n))))


def _concat(a: Number, b: Number) -> Number:
    """Concatenate decimal representations: concat(12, 34) = 1234."""
    return int(str(int(abs(a))) + str(int(abs(b))))


def _swap_result_digits(n: Number) -> Number:
    """Swap every pair of digits in the result, e.g. 46 → 64, 1234 → 2143."""
    s = str(abs(int(n)))
    out = []
    for i in range(0, len(s) - 1, 2):
        out.append(s[i + 1])
        out.append(s[i])
    if len(s) % 2 == 1:
        out.append(s[-1])
    return int("".join(out))


# ---------------------------------------------------------------------------
# Operand transforms (applied to the *inputs* before the operator is run)
# ---------------------------------------------------------------------------

# Each transform is: name → Callable[[list[Number]], list[Number]]

def _t_identity(inputs: list[Number]) -> list[Number]:
    return list(inputs)


def _t_reverse(inputs: list[Number]) -> list[Number]:
    return [_reverse_digits(x) for x in inputs]


def _t_reverse_order(inputs: list[Number]) -> list[Number]:
    """Swap the order of the two inputs (only meaningful for binary ops)."""
    return list(reversed(inputs))


def _t_reverse_both(inputs: list[Number]) -> list[Number]:
    """Reverse digits AND swap order of inputs."""
    return [_reverse_digits(x) for x in reversed(inputs)]


_TRANSFORMS: list[tuple[str, Callable[[list[Number]], list[Number]]]] = [
    ("AB_CD",       _t_identity),
    ("BA_DC",       _t_reverse),
    ("swap_order",  _t_reverse_order),
    ("rev_swap",    _t_reverse_both),
]


# ---------------------------------------------------------------------------
# Output post-transforms (applied to the *result* after the operator)
# ---------------------------------------------------------------------------

# After computing op(transformed_inputs), we may also try digit-swapping the result.
_OUT_TRANSFORMS: list[tuple[str, Callable[[Number], Number]]] = [
    ("",        lambda x: x),
    ("→YX",     _swap_result_digits),
]


# ---------------------------------------------------------------------------
# Operator library (binary)
# ---------------------------------------------------------------------------

def _safe_floordiv(a: Number, b: Number) -> Number:
    if b == 0:
        raise ZeroDivisionError
    return int(a) // int(b)


def _safe_mod(a: Number, b: Number) -> Number:
    if b == 0:
        raise ZeroDivisionError
    return int(a) % int(b)


def _safe_pow(a: Number, b: Number) -> Number:
    if abs(b) > 20:
        raise ValueError("exponent too large")
    return a ** b


def _safe_gcd(a: Number, b: Number) -> Number:
    return math.gcd(int(abs(a)), int(abs(b)))


def _safe_lcm(a: Number, b: Number) -> Number:
    g = _safe_gcd(a, b)
    if g == 0:
        return 0
    result = abs(int(a) * int(b)) // g
    if result > 10 ** 15:
        raise OverflowError
    return result


def _safe_lshift(a: Number, b: Number) -> Number:
    if b < 0 or b > 30:
        raise ValueError
    return int(a) << int(b)


def _safe_rshift(a: Number, b: Number) -> Number:
    if b < 0 or b > 30:
        raise ValueError
    return int(a) >> int(b)


# (name, fn)  — order reflects rough frequency in competition data
_BINARY_OPS: list[tuple[str, Callable[[Number, Number], Number]]] = [
    ("a+b",         lambda a, b: a + b),
    ("a-b",         lambda a, b: a - b),
    ("b-a",         lambda a, b: b - a),
    ("a*b",         lambda a, b: a * b),
    ("a//b",        _safe_floordiv),
    ("b//a",        lambda a, b: _safe_floordiv(b, a)),
    ("a%b",         _safe_mod),
    ("b%a",         lambda a, b: _safe_mod(b, a)),
    ("a**b",        _safe_pow),
    ("b**a",        lambda a, b: _safe_pow(b, a)),
    ("max(a,b)",    max),
    ("min(a,b)",    min),
    ("gcd(a,b)",    _safe_gcd),
    ("lcm(a,b)",    _safe_lcm),
    ("a&b",         lambda a, b: int(a) & int(b)),
    ("a|b",         lambda a, b: int(a) | int(b)),
    ("a^b",         lambda a, b: int(a) ^ int(b)),
    ("a<<b",        _safe_lshift),
    ("b<<a",        lambda a, b: _safe_lshift(b, a)),
    ("a>>b",        _safe_rshift),
    ("b>>a",        lambda a, b: _safe_rshift(b, a)),
    ("dsum(a)+dsum(b)", lambda a, b: _digit_sum(a) + _digit_sum(b)),
    ("dsum(a)*dsum(b)", lambda a, b: _digit_sum(a) * _digit_sum(b)),
    ("dsum(a)-dsum(b)", lambda a, b: _digit_sum(a) - _digit_sum(b)),
    ("concat(a,b)", lambda a, b: _concat(a, b)),
    ("concat(b,a)", lambda a, b: _concat(b, a)),
    ("a+dsum(b)",   lambda a, b: a + _digit_sum(b)),
    ("a*dsum(b)",   lambda a, b: a * _digit_sum(b)),
    ("b+dsum(a)",   lambda a, b: b + _digit_sum(a)),
    ("b*dsum(a)",   lambda a, b: b * _digit_sum(a)),
    ("(a+b)//2",    lambda a, b: (a + b) // 2),
    ("a*b+a",       lambda a, b: a * b + a),
    ("a*b-a",       lambda a, b: a * b - a),
]


# ---------------------------------------------------------------------------
# Operator library (unary)
# ---------------------------------------------------------------------------

def _safe_factorial(n: Number) -> Number:
    n = int(n)
    if n < 0 or n > 12:
        raise ValueError("factorial out of range")
    return math.factorial(n)


_UNARY_OPS: list[tuple[str, Callable[[Number], Number]]] = [
    ("abs(a)",      abs),
    ("-a",          lambda a: -a),
    ("2*a",         lambda a: 2 * a),
    ("a*a",         lambda a: a * a),
    ("a+1",         lambda a: a + 1),
    ("a-1",         lambda a: a - 1),
    ("a//2",        lambda a: int(a) // 2),
    ("dsum(a)",     _digit_sum),
    ("reverse(a)",  _reverse_digits),
    ("a!",          _safe_factorial),
]


# ---------------------------------------------------------------------------
# Core search
# ---------------------------------------------------------------------------

def _num_close(a: Number, b: Number) -> bool:
    """True if two numbers are equal within floating-point tolerance."""
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
    except (TypeError, OverflowError):
        return False


def _try_rule(rule: Rule, examples: list[Example]) -> bool:
    """Return True only if the rule produces the correct output for every example."""
    for ex in examples:
        try:
            t_inputs = _t_identity(ex.inputs)  # will be replaced below
            # Actually: rule carries its transform via partial closure — see usage.
            computed = rule.op_fn(*t_inputs)
            if not _num_close(computed, ex.output):
                return False
        except Exception:
            return False
    return True


@dataclass
class FoundRule:
    transform_name: str
    out_transform_name: str
    op_name: str
    raw_op_fn: Callable
    in_transform_fn: Callable[[list[Number]], list[Number]]
    out_transform_fn: Callable[[Number], Number]

    def apply(self, inputs: list[Number]) -> Number:
        t_inputs = self.in_transform_fn(inputs)
        raw = self.raw_op_fn(*t_inputs)
        return self.out_transform_fn(raw)

    @property
    def description(self) -> str:
        parts = []
        if self.transform_name not in ("AB_CD", ""):
            parts.append(f"inputs={self.transform_name}")
        parts.append(self.op_name)
        if self.out_transform_name:
            parts.append(f"result{self.out_transform_name}")
        return ", ".join(parts) if parts else self.op_name


def _search_binary(examples: list[Example]) -> FoundRule | None:
    """Brute-force search over transforms × ops × out-transforms for binary inputs."""
    for t_name, t_fn in _TRANSFORMS:
        for op_name, op_fn in _BINARY_OPS:
            for ot_name, ot_fn in _OUT_TRANSFORMS:
                ok = True
                for ex in examples:
                    if len(ex.inputs) < 2:
                        ok = False
                        break
                    try:
                        t_ins = t_fn(ex.inputs[:2])
                        raw = op_fn(t_ins[0], t_ins[1])
                        computed = ot_fn(raw)
                        if not _num_close(computed, ex.output):
                            ok = False
                            break
                    except Exception:
                        ok = False
                        break
                if ok:
                    return FoundRule(
                        transform_name=t_name,
                        out_transform_name=ot_name,
                        op_name=op_name,
                        raw_op_fn=op_fn,
                        in_transform_fn=t_fn,
                        out_transform_fn=ot_fn,
                    )
    return None


def _search_unary(examples: list[Example]) -> FoundRule | None:
    """Brute-force search over transforms × unary ops for single-input examples."""
    for t_name, t_fn in _TRANSFORMS:
        for op_name, op_fn in _UNARY_OPS:
            for ot_name, ot_fn in _OUT_TRANSFORMS:
                ok = True
                for ex in examples:
                    if len(ex.inputs) < 1:
                        ok = False
                        break
                    try:
                        t_ins = t_fn(ex.inputs[:1])
                        raw = op_fn(t_ins[0])
                        computed = ot_fn(raw)
                        if not _num_close(computed, ex.output):
                            ok = False
                            break
                    except Exception:
                        ok = False
                        break
                if ok:
                    return FoundRule(
                        transform_name=t_name,
                        out_transform_name=ot_name,
                        op_name=op_name,
                        raw_op_fn=op_fn,
                        in_transform_fn=t_fn,
                        out_transform_fn=ot_fn,
                    )
    return None


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class EquationSolver(BaseSolver):
    """Solves equation_numeric_deduce and equation_numeric_guess puzzles."""

    category = "equation"

    def solve(self, puzzle: dict) -> SolverResult:
        puzzle_id = puzzle.get("id", "")
        prompt: str = puzzle.get("prompt", "")
        ground_truth: str = puzzle.get("answer", "")

        examples, target_inputs = _parse_examples(prompt)

        # If we can't parse even one example, bail out early.
        if not examples:
            return SolverResult(
                puzzle_id=puzzle_id,
                category=self.category,
                predicted_answer="",
                is_correct=False,
                solve_method="no_examples_parsed",
                confidence=0.0,
            )

        arity = len(examples[0].inputs)

        # ------------------------------------------------------------------ #
        # Search for a rule that fits ALL examples                            #
        # ------------------------------------------------------------------ #
        found: FoundRule | None = None
        if arity >= 2:
            found = _search_binary(examples)
        if found is None:
            found = _search_unary(examples)

        # ------------------------------------------------------------------ #
        # Apply rule to target                                                #
        # ------------------------------------------------------------------ #
        predicted = ""
        solve_method = "no_rule_found"
        confidence = 0.0

        if found is not None and target_inputs is not None:
            try:
                result_num = found.apply(target_inputs)
                # Format as int when possible
                if isinstance(result_num, float) and result_num == int(result_num):
                    result_num = int(result_num)
                predicted = str(result_num)
                solve_method = found.description
                confidence = 1.0
            except Exception:
                predicted = ""
                solve_method = "rule_apply_error"
                confidence = 0.0
        elif found is None:
            # Fallback: return the most common output in the examples
            from collections import Counter
            counts = Counter(str(ex.output) for ex in examples)
            predicted = counts.most_common(1)[0][0] if counts else ""
            solve_method = "fallback_most_common_output"
            confidence = 0.1

        # ------------------------------------------------------------------ #
        # Verify                                                              #
        # ------------------------------------------------------------------ #
        is_correct = False
        if predicted and ground_truth:
            from src.metrics.competition import verify
            is_correct = verify(ground_truth, predicted)

        return SolverResult(
            puzzle_id=puzzle_id,
            category=self.category,
            predicted_answer=predicted,
            is_correct=is_correct,
            solve_method=solve_method,
            confidence=confidence,
        )


# ---------------------------------------------------------------------------
# Public helpers (imported by trace generator)
# ---------------------------------------------------------------------------

def parse_examples_and_target(
    prompt: str,
) -> tuple[list[Example], list[Number] | None]:
    return _parse_examples(prompt)


def find_rule(examples: list[Example]) -> FoundRule | None:
    if not examples:
        return None
    arity = len(examples[0].inputs)
    found: FoundRule | None = None
    if arity >= 2:
        found = _search_binary(examples)
    if found is None:
        found = _search_unary(examples)
    return found


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    import sys

    from src.metrics.competition import verify as _verify

    parser = argparse.ArgumentParser(
        description="Evaluate EquationSolver on classified puzzles."
    )
    parser.add_argument("--input", required=True, help="Path to puzzles_classified.jsonl")
    parser.add_argument("--verbose", action="store_true", help="Print wrong predictions.")
    args = parser.parse_args()

    solver = EquationSolver()
    total = correct = 0

    with open(args.input) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            puzzle = json.loads(line)
            cat = puzzle.get("category", "")
            if not cat.startswith("equation"):
                continue
            result = solver.solve(puzzle)
            total += 1
            if _verify(puzzle["answer"], result.predicted_answer):
                correct += 1
            elif args.verbose:
                print(
                    f"WRONG  id={puzzle.get('id', '?')} "
                    f"method={result.solve_method!r} "
                    f"predicted={result.predicted_answer!r} "
                    f"expected={puzzle['answer']!r}",
                    file=sys.stderr,
                )

    if total == 0:
        print("No equation puzzles found.")
    else:
        accuracy = correct / total
        print(f"Equation accuracy: {correct}/{total} = {accuracy:.4f}")
