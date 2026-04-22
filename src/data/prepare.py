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

def _count_equation_examples(prompt: str) -> int:
    """Count example lines containing '=' in a transformation-rules prompt."""
    count = 0
    for line in prompt.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Skip the question line and header lines
        if 'now,' in line.lower() or 'determine' in line.lower():
            continue
        if 'wonderland' in line.lower() or 'transformation' in line.lower():
            continue
        if '=' in line:
            count += 1
    return count


def classify_puzzle(prompt: str, answer: str) -> str:
    """
    Classify a puzzle into one of the defined categories.

    Uses clear text markers from the competition prompts:
    - "bit manipulation rule" → bit_manipulation
    - "gravitational constant" → gravity
    - "unit conversion" → unit_conversion
    - "numeral system" → numeral
    - "encryption rules" → cipher
    - "transformation rules is applied to equations" → equation or cryptarithm
    """
    # All prompts start with "In Alice's Wonderland, a secret ..." with
    # clear category indicators in the prompt text.

    if 'bit manipulation' in prompt:
        return 'bit_manipulation'

    if 'gravitational' in prompt:
        return 'gravity'

    if 'unit conversion' in prompt:
        return 'unit_conversion'

    if 'numeral system' in prompt or 'numeral' in prompt.lower():
        return 'numeral'

    if 'encryption rules' in prompt:
        return 'cipher'

    if 'transformation rules' in prompt and 'equations' in prompt:
        # Split into equation (numeric answer) vs cryptarithm (symbol answer)
        answer_stripped = answer.strip()
        is_numeric = bool(re.fullmatch(r'-?\d+(?:\.\d+)?', answer_stripped))

        # Count examples to split deduce vs guess
        n_examples = _count_equation_examples(prompt)

        if is_numeric:
            if n_examples >= 4:
                return 'equation_numeric_deduce'
            return 'equation_numeric_guess'
        else:
            if n_examples >= 4:
                return 'cryptarithm_deduce'
            return 'cryptarithm_guess'

    # Fallback heuristics for any edge cases
    answer_stripped = answer.strip()
    if re.fullmatch(r'[01]{8}', answer_stripped):
        return 'bit_manipulation'
    if _is_roman(answer_stripped):
        return 'numeral'

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

    # ------------------------------------------------------------------
    # Validation gate
    # ------------------------------------------------------------------
    errors: list[str] = []

    if len(records) == 0:
        errors.append("FATAL: zero puzzles classified")

    REQUIRED_CATEGORIES = {
        "numeral", "gravity", "unit_conversion", "cipher", "bit_manipulation",
    }
    missing = REQUIRED_CATEGORIES - set(counter.keys())
    if missing:
        errors.append(f"FATAL: missing required categories: {missing}")

    for cat, count in counter.items():
        if count == 0:
            errors.append(f"WARNING: category '{cat}' has 0 puzzles")

    # Spot-check: every record has required fields
    REQUIRED_FIELDS = {"id", "prompt", "answer", "category"}
    for i, rec in enumerate(records[:100]):  # check first 100
        missing_fields = REQUIRED_FIELDS - set(rec.keys())
        if missing_fields:
            errors.append(
                f"Record {i} missing fields: {missing_fields}"
            )
            break

    if errors:
        print("\n=== VALIDATION FAILED ===")
        for e in errors:
            print(f"  {e}")
        if any(e.startswith("FATAL") for e in errors):
            sys.exit(1)
    else:
        print("\n=== VALIDATION PASSED ===")
        print(f"  {len(records)} puzzles across {len(counter)} categories")


if __name__ == '__main__':
    main()
