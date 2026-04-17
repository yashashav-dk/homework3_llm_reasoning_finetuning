"""
prepare.py — classify puzzles from train.csv into categories.

Usage:
    python prepare.py --input data/train.csv --output data/puzzles_classified.jsonl
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ROMAN_CHARS_RE = re.compile(r'^[IVXLCDMivxlcdm]+$')
_BINARY8_RE = re.compile(r'\b[01]{8}\b')
_NUMBER_1_100_RE = re.compile(r'^\s*\d{1,3}\s*$')

# Detects a stand-alone 8-bit binary token
_BINARY8_STRICT_RE = re.compile(r'(?<!\d)[01]{8}(?!\d)')


def _is_roman(s: str) -> bool:
    """Return True if *s* looks like a Roman numeral string (non-empty, only Roman chars)."""
    s = s.strip()
    return bool(s) and bool(_ROMAN_CHARS_RE.match(s))


def _count_binary8_tokens(text: str) -> int:
    """Count 8-bit binary tokens in text."""
    return len(_BINARY8_STRICT_RE.findall(text))


def _count_numeric_io_pairs(prompt: str) -> int:
    """
    Count lines/segments that look like  <number> → <number>  (or  → / => / : ).
    Heuristic: find occurrences of  'number <sep> number'  where sep is →, ->, =>, :, etc.
    """
    pair_re = re.compile(
        r'(?:^|\n|,|;)\s*-?\d+(?:\.\d+)?\s*(?:→|->|=>|:|\|)\s*-?\d+(?:\.\d+)?',
        re.MULTILINE,
    )
    return len(pair_re.findall(prompt))


def _count_word_pairs(prompt: str) -> int:
    """
    Count lines/segments that look like  <word> → <word>.
    """
    pair_re = re.compile(
        r'(?:^|\n|,|;)\s*[A-Za-z]+\s*(?:→|->|=>|:|\|)\s*[A-Za-z]+',
        re.MULTILINE,
    )
    return len(pair_re.findall(prompt))


def _count_letter_digit_pairs(prompt: str) -> int:
    """
    Count letter=digit or letter->digit assignments (cryptarithm hints).
    """
    pair_re = re.compile(r'[A-Za-z]\s*(?:=|->|:)\s*\d')
    return len(pair_re.findall(prompt))


def _count_arithmetic_equations_with_letters(prompt: str) -> int:
    """
    Detect things like SEND + MORE = MONEY, i.e. multi-letter tokens joined by +/-/* with =.
    """
    eq_re = re.compile(
        r'[A-Z]{2,}\s*[+\-*/]\s*[A-Z]{2,}\s*=\s*[A-Z]{2,}',
        re.IGNORECASE,
    )
    return len(eq_re.findall(prompt))


def _has_time_distance_context(prompt: str) -> bool:
    """Return True if the prompt looks like a gravity / free-fall problem."""
    keywords = ['gravity', 'gravit', 'falling', 'free fall', 'freefall',
                'distance', 'drop', 'g =', 'g=', '9.8', '9.81', 'acceleration']
    low = prompt.lower()
    return any(kw in low for kw in keywords)


def _has_unit_conversion_context(prompt: str) -> bool:
    """Return True if prompt looks like a unit-conversion problem."""
    keywords = ['convert', 'conversion', 'unit', 'meter', 'feet', 'foot',
                'mile', 'kilomet', 'kilogram', 'pound', 'celsius', 'fahrenheit',
                'liter', 'gallon', 'inch', 'centimeter', 'yard']
    low = prompt.lower()
    return any(kw in low for kw in keywords)


def _has_cipher_context(prompt: str) -> bool:
    """Return True if prompt looks like a cipher/encryption problem."""
    keywords = ['encrypt', 'decrypt', 'cipher', 'secret', 'encode', 'decode',
                'substitut', 'shift', 'coded', 'plaintext', 'ciphertext']
    low = prompt.lower()
    return any(kw in low for kw in keywords)


def _has_roman_numeral_context(prompt: str) -> bool:
    """Return True if prompt explicitly mentions Roman numerals."""
    low = prompt.lower()
    return 'roman' in low or 'numeral' in low


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

def classify_puzzle(prompt: str, answer: str) -> str:
    """
    Classify a puzzle into one of the defined categories.

    Parameters
    ----------
    prompt : str
        The puzzle prompt text.
    answer : str
        The expected answer string.

    Returns
    -------
    str
        One of: numeral, gravity, unit_conversion, cipher, bit_manipulation,
        equation_numeric_deduce, equation_numeric_guess,
        cryptarithm_deduce, cryptarithm_guess.
    """
    answer_stripped = answer.strip()

    # ------------------------------------------------------------------
    # 1. bit_manipulation — 8-bit binary strings
    # ------------------------------------------------------------------
    if _BINARY8_STRICT_RE.fullmatch(answer_stripped) or re.fullmatch(r'[01]{8}', answer_stripped):
        # Confirm prompt also has binary-looking content
        if _count_binary8_tokens(prompt) >= 1:
            return 'bit_manipulation'

    # Check prompt heavily loaded with 8-bit binary tokens even if answer
    # doesn't look binary (edge case safety net)
    if _count_binary8_tokens(prompt) >= 3:
        return 'bit_manipulation'

    # ------------------------------------------------------------------
    # 2. numeral — Roman numeral conversion
    # ------------------------------------------------------------------
    if _has_roman_numeral_context(prompt):
        return 'numeral'

    # Answer is a pure Roman numeral string
    if _is_roman(answer_stripped) and len(answer_stripped) >= 1:
        return 'numeral'

    # Answer is a plain number 1-100 AND prompt has Roman context (secondary check)
    if _NUMBER_1_100_RE.match(answer_stripped):
        num = int(answer_stripped.strip())
        if 1 <= num <= 100 and _has_roman_numeral_context(prompt):
            return 'numeral'

    # ------------------------------------------------------------------
    # 3. gravity — physics d = 0.5 * g * t^2
    # ------------------------------------------------------------------
    if _has_time_distance_context(prompt):
        return 'gravity'

    # ------------------------------------------------------------------
    # 4. unit_conversion — linear scaling
    # ------------------------------------------------------------------
    if _has_unit_conversion_context(prompt):
        # Require some numeric pairs as evidence
        if _count_numeric_io_pairs(prompt) >= 1:
            return 'unit_conversion'

    # ------------------------------------------------------------------
    # 5. cipher — substitution cipher on text/words
    # ------------------------------------------------------------------
    if _has_cipher_context(prompt):
        return 'cipher'

    # Word→word pairs without obvious arithmetic → likely cipher
    word_pairs = _count_word_pairs(prompt)
    if word_pairs >= 2 and not re.search(r'\d', answer_stripped):
        return 'cipher'

    # ------------------------------------------------------------------
    # 6. cryptarithm — letter-to-digit substitution arithmetic
    # ------------------------------------------------------------------
    letter_assignments = _count_letter_digit_pairs(prompt)
    letter_equations = _count_arithmetic_equations_with_letters(prompt)

    if letter_equations >= 1 or letter_assignments >= 3:
        # Distinguish deduce vs guess by number of constraints
        if letter_assignments >= 3 or letter_equations >= 2:
            return 'cryptarithm_deduce'
        return 'cryptarithm_guess'

    # ------------------------------------------------------------------
    # 7. equation_numeric — arithmetic operator discovery
    # ------------------------------------------------------------------
    numeric_pairs = _count_numeric_io_pairs(prompt)

    if numeric_pairs >= 1:
        if numeric_pairs >= 3:
            return 'equation_numeric_deduce'
        return 'equation_numeric_guess'

    # ------------------------------------------------------------------
    # Fallback: try unit_conversion even without confirmed pairs
    # ------------------------------------------------------------------
    if _has_unit_conversion_context(prompt):
        return 'unit_conversion'

    # Default: if answer is all digits, treat as equation guess
    if re.fullmatch(r'-?\d+(?:\.\d+)?', answer_stripped):
        return 'equation_numeric_guess'

    # Last resort: cipher (unrecognised text transformation)
    return 'cipher'


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _read_csv(path: str):
    """Yield dicts with keys id, prompt, answer from a CSV file."""
    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield {
                'id': row.get('id', ''),
                'prompt': row.get('prompt', ''),
                'answer': row.get('answer', ''),
            }


def _write_jsonl(records, path: str) -> None:
    """Write records (list of dicts) as newline-delimited JSON."""
    with open(path, 'w', encoding='utf-8') as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')


def _print_summary(counter: Counter) -> None:
    """Print a formatted summary table of category counts."""
    total = sum(counter.values())
    print()
    print(f"{'Category':<35} {'Count':>7}  {'%':>6}")
    print('-' * 52)
    for category, count in sorted(counter.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / total if total else 0.0
        print(f"{category:<35} {count:>7}  {pct:>5.1f}%")
    print('-' * 52)
    print(f"{'TOTAL':<35} {total:>7}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Classify puzzles from train.csv into categories.',
    )
    parser.add_argument(
        '--input',
        default='data/train.csv',
        help='Path to input CSV file (default: data/train.csv)',
    )
    parser.add_argument(
        '--output',
        default='data/puzzles_classified.jsonl',
        help='Path to output JSONL file (default: data/puzzles_classified.jsonl)',
    )
    args = parser.parse_args()

    records = []
    counter: Counter = Counter()

    print(f"Reading puzzles from: {args.input}")
    try:
        rows = list(_read_csv(args.input))
    except FileNotFoundError:
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    for row in rows:
        category = classify_puzzle(row['prompt'], row['answer'])
        rec = {
            'id': row['id'],
            'prompt': row['prompt'],
            'answer': row['answer'],
            'category': category,
        }
        records.append(rec)
        counter[category] += 1

    _write_jsonl(records, args.output)
    print(f"Written {len(records)} classified puzzles to: {args.output}")

    _print_summary(counter)


if __name__ == '__main__':
    main()
