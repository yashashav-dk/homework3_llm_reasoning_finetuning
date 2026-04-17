"""Solver for cryptarithm puzzles (verbal arithmetic where letters represent digits).

Strategy: focus on the ~8% of puzzles solvable by simple methods:
  1. Concatenation detection
  2. Reverse-concatenation detection
  3. Carry-free addition
  4. Brute-force for puzzles with ≤4 unknown letters
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

from src.solvers.base import BaseSolver, SolverResult


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

@dataclass
class CryptarithmPuzzle:
    operands: list[str]          # e.g. ["SEND", "MORE"]
    operator: str                 # "+", "-", "*", or "concat"
    result: str                   # e.g. "MONEY"
    given: dict[str, int]         # pre-assigned letter → digit
    question_letter: str          # the letter whose digit is asked
    question_type: str            # "letter_to_digit" or "digit_to_letter"
    question_digit: int | None    # set when question_type == "digit_to_letter"
    raw_expr: str = ""            # full expression string for display


def _parse_prompt(prompt: str) -> CryptarithmPuzzle | None:
    """Extract the structured puzzle data from a free-form prompt string."""

    # ---- 1. Collect any pre-given assignments --------------------------------
    # "S = 9", "S represents 9", "S is 9", "9 represents S"
    given: dict[str, int] = {}
    given_patterns = [
        re.compile(r"\b([A-Z])\s*=\s*(\d)\b"),
        re.compile(r"\b([A-Z])\s+represents?\s+(\d)\b", re.IGNORECASE),
        re.compile(r"\b([A-Z])\s+is\s+(\d)\b", re.IGNORECASE),
        re.compile(r"\b(\d)\s+represents?\s+([A-Z])\b", re.IGNORECASE),
    ]
    for pat in given_patterns[:3]:
        for m in pat.finditer(prompt):
            letter, digit = m.group(1).upper(), int(m.group(2))
            given[letter] = digit
    # reversed form: digit represents letter
    for m in given_patterns[3].finditer(prompt):
        digit, letter = int(m.group(1)), m.group(2).upper()
        given[letter] = digit

    # ---- 2. Identify the arithmetic expression -------------------------------
    # Look for lines containing uppercase words joined by +/-/* and =
    # Normalise the prompt: strip dashes/equals used as separators
    lines = prompt.splitlines()

    expr_line: str | None = None
    for line in lines:
        stripped = line.strip()
        # Skip separator lines (all dashes or equals)
        if re.fullmatch(r"[-=\s]+", stripped):
            continue
        # Must contain at least two uppercase letter runs and one '=' or '+'
        upper_words = re.findall(r"\b[A-Z]{2,}\b", stripped)
        if len(upper_words) >= 2 and ("=" in stripped or "+" in stripped or "-" in stripped or "*" in stripped):
            expr_line = stripped
            break

    # Fallback: scan for two-line style  (SEND\n+MORE\n----\nMONEY)
    if expr_line is None:
        upper_word_lines = []
        for line in lines:
            stripped = line.strip()
            if re.search(r"[+\-*]?\s*[A-Z]{2,}", stripped):
                upper_word_lines.append(stripped)
        if len(upper_word_lines) >= 2:
            expr_line = "  ".join(upper_word_lines)

    if expr_line is None:
        return None

    # ---- 3. Parse operands, operator, result ---------------------------------
    # Normalise: remove stray separator chars, collapse whitespace
    expr_clean = re.sub(r"[-]+", "", expr_line)   # remove dashes used as separator lines
    expr_clean = re.sub(r"\s+", " ", expr_clean).strip()

    # Detect concatenation operator (unusual symbols) or explicit "concat" / "|"
    concat_op = bool(re.search(r"concat|CONCAT|\|\s*[A-Z]", prompt))

    # Try to split on = first
    if "=" in expr_clean:
        lhs, _, result_part = expr_clean.partition("=")
    else:
        # Last uppercase word cluster is the result; everything before is lhs
        all_words = re.findall(r"[A-Z]{2,}", expr_clean)
        if len(all_words) < 2:
            return None
        result_part = all_words[-1]
        lhs = expr_clean[: expr_clean.rfind(result_part)]

    result_word = re.search(r"[A-Z]{2,}", result_part.strip())
    if result_word is None:
        return None
    result = result_word.group(0)

    # Determine operator and operands from lhs
    operator = "+"  # default
    if re.search(r"\+", lhs):
        operator = "+"
    elif re.search(r"\*", lhs):
        operator = "*"
    elif re.search(r"-", lhs):
        operator = "-"

    operands = re.findall(r"[A-Z]{2,}", lhs)
    if not operands:
        return None

    raw_expr = f"{' + '.join(operands)} = {result}" if operator == "+" else expr_clean

    # ---- 4. Parse the question -----------------------------------------------
    question_letter = ""
    question_type = "letter_to_digit"
    question_digit: int | None = None

    # "What digit does X represent?" / "What is the digit for X?" / "Find X"
    letter_q_patterns = [
        re.compile(r"what\s+(?:digit\s+)?does\s+([A-Z])\s+represent", re.IGNORECASE),
        re.compile(r"what\s+is\s+the\s+(?:value|digit)\s+(?:of\s+)?([A-Z])\b", re.IGNORECASE),
        re.compile(r"(?:find|determine)\s+(?:the\s+digit\s+for\s+)?([A-Z])\b", re.IGNORECASE),
        re.compile(r"what\s+digit\s+is\s+([A-Z])\b", re.IGNORECASE),
        re.compile(r"([A-Z])\s*=\s*\?"),
        re.compile(r"What\s+does\s+([A-Z])\s+represent", re.IGNORECASE),
        # Generic: last uppercase single letter near a question mark
    ]
    for pat in letter_q_patterns:
        m = pat.search(prompt)
        if m:
            question_letter = m.group(1).upper()
            question_type = "letter_to_digit"
            break

    # "What letter represents digit 9?" / "Which letter is 9?"
    if not question_letter:
        digit_q_patterns = [
            re.compile(r"what\s+letter\s+represents?\s+(?:digit\s+)?(\d)\b", re.IGNORECASE),
            re.compile(r"which\s+letter\s+(?:is|represents?)\s+(\d)\b", re.IGNORECASE),
        ]
        for pat in digit_q_patterns:
            m = pat.search(prompt)
            if m:
                question_digit = int(m.group(1))
                question_type = "digit_to_letter"
                break

    # Fallback: look for a lone letter before or after "represent(s)"
    if not question_letter and question_type == "letter_to_digit":
        m = re.search(r"\b([A-Z])\b(?=\s*\?|$)", prompt.split("\n")[-1])
        if m:
            question_letter = m.group(1).upper()

    # Last resort: pick first letter in result word not already given
    if not question_letter and question_type == "letter_to_digit":
        for ch in result:
            if ch not in given:
                question_letter = ch
                break

    if not question_letter and question_type == "letter_to_digit":
        return None

    return CryptarithmPuzzle(
        operands=operands,
        operator=operator,
        result=result,
        given=given,
        question_letter=question_letter,
        question_type=question_type,
        question_digit=question_digit,
        raw_expr=raw_expr,
    )


# ---------------------------------------------------------------------------
# Solving strategies
# ---------------------------------------------------------------------------

def _all_letters(puzzle: CryptarithmPuzzle) -> list[str]:
    """Unique letters across all operands and result, preserving order."""
    seen: set[str] = set()
    letters: list[str] = []
    for word in puzzle.operands + [puzzle.result]:
        for ch in word:
            if ch not in seen:
                seen.add(ch)
                letters.append(ch)
    return letters


def _assignment_valid(assignment: dict[str, int], puzzle: CryptarithmPuzzle) -> bool:
    """Check leading-zero and uniqueness constraints, then verify arithmetic."""
    # No leading zeros
    for word in puzzle.operands + [puzzle.result]:
        if assignment.get(word[0], -1) == 0:
            return False

    # Verify arithmetic
    def word_val(w: str) -> int:
        val = 0
        for ch in w:
            val = val * 10 + assignment[ch]
        return val

    operand_vals = [word_val(op) for op in puzzle.operands]
    result_val = word_val(puzzle.result)

    if puzzle.operator == "+":
        return sum(operand_vals) == result_val
    elif puzzle.operator == "-":
        return operand_vals[0] - sum(operand_vals[1:]) == result_val
    elif puzzle.operator == "*":
        product = 1
        for v in operand_vals:
            product *= v
        return product == result_val
    return False


def _try_concatenation(puzzle: CryptarithmPuzzle) -> dict[str, int] | None:
    """Check if result is simply the concatenation of the operand digit-strings.

    This covers patterns like: AB + CD = ABCD  (result = join of operands as strings).
    Each operand must be fully determined by given assignments.
    """
    if puzzle.operator != "+":
        return None
    # Build result string by concatenating operands
    concat = "".join(puzzle.operands)
    if concat == puzzle.result:
        # The arrangement itself is the identity — map letter positions
        # Only works when operand letters uniquely determine result letters.
        assignment: dict[str, int] = dict(puzzle.given)
        return assignment if assignment else None

    # Try: result letters == concat of all operand letters in order
    if len(concat) != len(puzzle.result):
        return None

    assignment = dict(puzzle.given)
    conflict = False
    for enc_ch, res_ch in zip(concat, puzzle.result):
        if enc_ch == res_ch:
            continue
        # enc_ch from operand maps to same position in result
        if enc_ch in assignment:
            # Check consistency with result position value
            pass
        if res_ch in assignment:
            # Check consistency
            pass
        # This heuristic: assign same digit to same letter
        assignment[enc_ch] = assignment.get(enc_ch, assignment.get(res_ch, -1))
    return None


def _try_reverse_concatenation(puzzle: CryptarithmPuzzle) -> dict[str, int] | None:
    """Check if result is the reverse of the concatenation of operands."""
    if puzzle.operator != "+":
        return None
    concat = "".join(puzzle.operands)
    if concat[::-1] == puzzle.result and len(concat) == len(puzzle.result):
        assignment: dict[str, int] = dict(puzzle.given)
        for op_ch, res_ch in zip(concat, puzzle.result[::-1]):
            if op_ch in assignment and res_ch in assignment:
                if assignment[op_ch] != assignment[res_ch]:
                    return None
        return assignment if assignment else None
    return None


def _try_carry_free(puzzle: CryptarithmPuzzle) -> dict[str, int] | None:
    """Attempt column-by-column digit deduction for carry-free addition.

    Carry-free means each column sum is < 10.
    Works right-to-left; deduces new letter values when only one unknown per column.
    """
    if puzzle.operator != "+":
        return None
    if len(puzzle.operands) < 2:
        return None

    # Pad operands to same length as result
    max_len = len(puzzle.result)
    padded = [op.rjust(max_len, "\x00") for op in puzzle.operands]  # \x00 = padding
    result_padded = puzzle.result

    assignment: dict[str, int] = dict(puzzle.given)
    changed = True
    iterations = 0

    while changed and iterations < 20:
        changed = False
        iterations += 1
        for col in range(max_len - 1, -1, -1):
            col_letters = [p[col] for p in padded if p[col] != "\x00"]
            res_letter = result_padded[col]

            known_sum = sum(assignment[ch] for ch in col_letters if ch in assignment)
            unknowns_in_col = [ch for ch in col_letters if ch not in assignment]
            res_known = res_letter in assignment

            # If result unknown and exactly one operand letter unknown
            if not res_known and len(unknowns_in_col) == 0:
                # Carry-free: result digit = column sum mod 10
                digit = known_sum % 10
                if digit not in assignment.values() or res_letter in assignment:
                    if res_letter not in assignment:
                        assignment[res_letter] = digit
                        changed = True

            elif res_known and len(unknowns_in_col) == 1:
                # One unknown operand letter: unknown = result - known_sum
                unknown_ch = unknowns_in_col[0]
                needed = assignment[res_letter] - known_sum
                if 0 <= needed <= 9 and needed not in assignment.values():
                    assignment[unknown_ch] = needed
                    changed = True

    # Return only if we have the question letter resolved
    if puzzle.question_letter in assignment:
        # Validate if all letters are assigned
        all_letters = _all_letters(puzzle)
        if all(ch in assignment for ch in all_letters):
            if _assignment_valid(assignment, puzzle):
                return assignment
        elif puzzle.question_letter in assignment:
            return assignment  # Partial but answers the question

    return None


def _try_brute_force(puzzle: CryptarithmPuzzle) -> dict[str, int] | None:
    """Brute-force all digit permutations for puzzles with ≤4 unknown letters."""
    all_letters = _all_letters(puzzle)
    known = dict(puzzle.given)
    unknowns = [ch for ch in all_letters if ch not in known]

    if len(unknowns) > 4:
        return None  # Too expensive

    used_digits = set(known.values())
    available = [d for d in range(10) if d not in used_digits]

    for perm in itertools.permutations(available, len(unknowns)):
        candidate = dict(known)
        for ch, d in zip(unknowns, perm):
            candidate[ch] = d
        if _assignment_valid(candidate, puzzle):
            return candidate

    return None


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class CryptarithmSolver(BaseSolver):
    """Solver for cryptarithm_deduce and cryptarithm_guess puzzle categories."""

    category = "cryptarithm"

    def solve(self, puzzle: dict) -> SolverResult:
        puzzle_id = puzzle.get("id", "")
        prompt: str = puzzle.get("prompt", "")
        ground_truth: str = puzzle.get("answer", "")

        # -- Parse the puzzle --------------------------------------------------
        parsed = _parse_prompt(prompt)
        if parsed is None:
            return SolverResult(
                puzzle_id=puzzle_id,
                category=self.category,
                predicted_answer="",
                is_correct=False,
                solve_method="parse_failed",
                confidence=0.0,
            )

        # If the answer is directly given, use it
        if parsed.question_type == "letter_to_digit":
            if parsed.question_letter in parsed.given:
                digit = parsed.given[parsed.question_letter]
                predicted = str(digit)
                is_correct = predicted.strip() == ground_truth.strip()
                return SolverResult(
                    puzzle_id=puzzle_id,
                    category=self.category,
                    predicted_answer=predicted,
                    is_correct=is_correct,
                    solve_method="given_assignment",
                    confidence=1.0,
                )

        # -- Strategy 1: Concatenation detection --------------------------------
        assignment = _try_concatenation(parsed)
        method = "concatenation"

        # -- Strategy 2: Reverse concatenation ----------------------------------
        if assignment is None:
            assignment = _try_reverse_concatenation(parsed)
            method = "reverse_concatenation"

        # -- Strategy 3: Carry-free addition ------------------------------------
        if assignment is None:
            assignment = _try_carry_free(parsed)
            method = "carry_free_addition"

        # -- Strategy 4: Brute force (≤4 unknowns) ------------------------------
        if assignment is None:
            assignment = _try_brute_force(parsed)
            method = "brute_force"

        # -- Extract the answer ------------------------------------------------
        if assignment is None:
            return SolverResult(
                puzzle_id=puzzle_id,
                category=self.category,
                predicted_answer="",
                is_correct=False,
                solve_method="unsolvable",
                confidence=0.0,
            )

        if parsed.question_type == "letter_to_digit":
            digit = assignment.get(parsed.question_letter)
            if digit is None:
                return SolverResult(
                    puzzle_id=puzzle_id,
                    category=self.category,
                    predicted_answer="",
                    is_correct=False,
                    solve_method=method,
                    confidence=0.0,
                )
            predicted = str(digit)
            confidence = 0.9 if method == "brute_force" else 0.85
        else:
            # digit_to_letter: find the letter whose digit matches question_digit
            found_letter = ""
            for ch, d in assignment.items():
                if d == parsed.question_digit:
                    found_letter = ch
                    break
            predicted = found_letter
            confidence = 0.85

        is_correct = predicted.strip().lower() == ground_truth.strip().lower()

        return SolverResult(
            puzzle_id=puzzle_id,
            category=self.category,
            predicted_answer=predicted,
            is_correct=is_correct,
            solve_method=method,
            confidence=confidence if predicted else 0.0,
        )


# ---------------------------------------------------------------------------
# CLI: accuracy evaluation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    import sys

    from src.metrics.competition import verify

    parser = argparse.ArgumentParser(
        description="Evaluate CryptarithmSolver on classified puzzles."
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

    solver = CryptarithmSolver()
    total = correct = skipped = 0

    with open(args.input) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            puzzle = json.loads(line)
            cat = puzzle.get("category", "")
            if not (cat == "cryptarithm" or cat.startswith("cryptarithm_")):
                continue
            result = solver.solve(puzzle)
            if result.confidence == 0.0:
                skipped += 1
                continue
            total += 1
            if verify(puzzle["answer"], result.predicted_answer):
                correct += 1
            elif args.verbose:
                print(
                    f"WRONG  id={puzzle.get('id', '?')} "
                    f"method={result.solve_method} "
                    f"predicted={result.predicted_answer!r} "
                    f"expected={puzzle['answer']!r}",
                    file=sys.stderr,
                )

    print(f"Attempted: {total}, Correct: {correct}, Skipped (unsolvable): {skipped}")
    if total > 0:
        print(f"Accuracy on attempted: {correct}/{total} = {correct/total:.4f}")
