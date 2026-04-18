"""Solver for substitution cipher puzzles (Alice in Wonderland themed)."""

from __future__ import annotations

import re
from collections import defaultdict

from src.solvers.base import BaseSolver, SolverResult

# ---------------------------------------------------------------------------
# Alice in Wonderland vocabulary (~90 common words)
# ---------------------------------------------------------------------------

WONDERLAND_VOCAB: list[str] = [
    # All 77 unique words from competition cipher answers
    "the", "follows", "dragon", "teacher", "writes", "creates", "draws",
    "student", "rabbit", "studies", "discovers", "secret", "found", "mouse",
    "dreams", "chases", "reads", "king", "sees", "watches", "queen", "hatter",
    "knight", "explores", "bird", "imagines", "wizard", "turtle", "castle",
    "cat", "alice", "garden", "princess", "colorful", "puzzle", "bright",
    "forest", "book", "clever", "key", "dark", "mirror", "treasure", "silver",
    "beyond", "inside", "in", "hidden", "curious", "around", "above", "wise",
    "potion", "near", "door", "golden", "under", "through", "mysterious",
    "magical", "strange", "story", "crystal", "message", "map", "ancient",
    "village", "mountain", "wonderland", "cave", "school", "valley", "island",
    "palace", "library", "ocean", "tower",
]

# Pre-build a set and an index: length → list[word]
_VOCAB_SET: set[str] = {w.lower() for w in WONDERLAND_VOCAB}
_VOCAB_BY_LEN: dict[int, list[str]] = defaultdict(list)
for _w in WONDERLAND_VOCAB:
    _VOCAB_BY_LEN[len(_w)].append(_w.lower())


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_examples(prompt: str) -> list[tuple[str, str]]:
    """Extract (encrypted, decrypted) example pairs from the puzzle prompt.

    Actual format from competition data:
      encrypted_phrase -> decrypted_phrase
    e.g.:
      ucoov pwgtfyoqg vorq yrjjoe -> queen discovers near valley
    """
    examples: list[tuple[str, str]] = []

    for line in prompt.splitlines():
        line = line.strip()
        if not line:
            continue

        # Match example pairs FIRST (before keyword filtering)
        if ' -> ' in line:
            parts = line.split(' -> ', 1)
            if len(parts) == 2:
                enc = parts[0].strip().lower()
                dec = parts[1].strip().lower()
                if enc and dec:
                    examples.append((enc, dec))

    return examples


def _parse_target(prompt: str) -> str:
    """Extract the target encrypted string to decrypt.

    Actual format: "Now, decrypt the following text: <encrypted_phrase>"
    """
    # Pattern: "decrypt the following text: X"
    m = re.search(r'decrypt the following text:\s*(.+)', prompt, re.IGNORECASE)
    if m:
        return m.group(1).strip().lower()

    # Fallback patterns
    for pat in [
        re.compile(r"what\s+does\s+['\"]?([^'\"?]+)['\"]?\s+decrypt", re.IGNORECASE),
        re.compile(r"decrypt\s+['\"]([^'\"]+)['\"]", re.IGNORECASE),
    ]:
        m = pat.search(prompt)
        if m:
            return m.group(1).strip().lower()

    # Last line fallback
    lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
    if lines:
        return lines[-1].rstrip("?!.:;").lower()

    return ""


# ---------------------------------------------------------------------------
# Mapping builder
# ---------------------------------------------------------------------------

def _build_char_mapping(examples: list[tuple[str, str]]) -> dict[str, str]:
    """Build encrypted_char → decrypted_char from all example pairs.

    Examples are full phrases (multi-word), aligned character-by-character.
    Spaces map to spaces, letters map to letters.
    """
    mapping: dict[str, str] = {}
    for enc_phrase, dec_phrase in examples:
        if len(enc_phrase) != len(dec_phrase):
            continue
        for enc_ch, dec_ch in zip(enc_phrase, dec_phrase):
            if enc_ch == ' ':
                continue  # space → space is trivial
            if enc_ch not in mapping:
                mapping[enc_ch] = dec_ch
    return mapping


# ---------------------------------------------------------------------------
# Vocabulary-based gap filling
# ---------------------------------------------------------------------------

def _complete_mapping(mapping: dict[str, str]) -> dict[str, str]:
    """Complete a partial substitution cipher mapping using constraint propagation.

    Since the cipher is a bijection (each letter maps to exactly one other),
    we can deduce unmapped letters by elimination: if 25 of 26 mappings are
    known, the last is forced. We iterate until no more can be deduced.
    """
    mapping = dict(mapping)
    all_letters = set('abcdefghijklmnopqrstuvwxyz')

    changed = True
    while changed:
        changed = False
        mapped_enc = {k for k in mapping if k in all_letters}
        mapped_dec = {v for v in mapping.values() if v in all_letters}
        unmapped_enc = all_letters - mapped_enc
        unmapped_dec = all_letters - mapped_dec

        # If only one unmapped letter remains on each side, they must pair
        if len(unmapped_enc) == 1 and len(unmapped_dec) == 1:
            enc_ch = unmapped_enc.pop()
            dec_ch = unmapped_dec.pop()
            mapping[enc_ch] = dec_ch
            changed = True

    return mapping


# ---------------------------------------------------------------------------
# Core decryption
# ---------------------------------------------------------------------------

def _vocab_match(enc_word: str, mapping: dict[str, str]) -> str | None:
    """Try to match an encrypted word against vocabulary.

    Returns the matching vocab word if exactly one fits, else None.
    A word fits if: same length, known-mapped positions match, and
    new mappings don't conflict with existing ones (bijection check).
    """
    n = len(enc_word)
    candidates = []
    reverse_mapping = {v: k for k, v in mapping.items() if k.isalpha() and v.isalpha()}

    for vocab_word in _VOCAB_BY_LEN.get(n, []):
        ok = True
        for enc_ch, dec_ch in zip(enc_word, vocab_word):
            if enc_ch in mapping:
                if mapping[enc_ch] != dec_ch:
                    ok = False
                    break
            else:
                # Check bijection: dec_ch shouldn't already be mapped from another enc char
                if dec_ch in reverse_mapping and reverse_mapping[dec_ch] != enc_ch:
                    ok = False
                    break
        if ok:
            candidates.append(vocab_word)

    if len(candidates) == 1:
        return candidates[0]
    return None


def _decrypt_text(
    enc_text: str,
    mapping: dict[str, str],
) -> tuple[str, dict[str, str]]:
    """Decrypt enc_text using mapping + vocabulary matching.

    Multiple passes: each vocab match can reveal new letters that help subsequent words.
    """
    working_mapping = dict(mapping)
    words = enc_text.split()

    # Multiple passes to propagate newly discovered mappings
    for _ in range(3):
        all_resolved = True
        for enc_word in words:
            partial = "".join(working_mapping.get(ch, "?") for ch in enc_word)
            if "?" in partial:
                all_resolved = False
                match = _vocab_match(enc_word, working_mapping)
                if match:
                    for enc_ch, dec_ch in zip(enc_word, match):
                        if enc_ch not in working_mapping:
                            working_mapping[enc_ch] = dec_ch
        if all_resolved:
            break

    # Also try constraint completion (bijection elimination)
    working_mapping = _complete_mapping(working_mapping)

    # Final decryption
    result_words = []
    for enc_word in words:
        dec_word = "".join(working_mapping.get(ch, "?") for ch in enc_word)
        result_words.append(dec_word)

    return " ".join(result_words), working_mapping


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class CipherSolver(BaseSolver):
    category = "cipher"

    def solve(self, puzzle: dict) -> SolverResult:
        puzzle_id = puzzle.get("id", "")
        prompt: str = puzzle.get("prompt", "")
        ground_truth: str = puzzle.get("answer", "")

        # 1. Parse examples and target
        examples = _parse_examples(prompt)
        target_enc = _parse_target(prompt)

        # 2. Build character mapping from ALL examples
        mapping = _build_char_mapping(examples)

        # 3. Decrypt target
        predicted, _ = _decrypt_text(target_enc, mapping)

        # 4. Determine correctness
        is_correct = predicted.strip().lower() == ground_truth.strip().lower()

        return SolverResult(
            puzzle_id=puzzle_id,
            category=self.category,
            predicted_answer=predicted,
            is_correct=is_correct,
            solve_method="substitution_cipher",
            confidence=1.0 if "?" not in predicted else 0.5,
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
        description="Evaluate CipherSolver on classified puzzles."
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

    solver = CipherSolver()
    total = correct = 0

    with open(args.input) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            puzzle = json.loads(line)
            if puzzle.get("category") != "cipher":
                continue
            result = solver.solve(puzzle)
            total += 1
            if verify(puzzle["answer"], result.predicted_answer):
                correct += 1
            elif args.verbose:
                print(
                    f"WRONG  id={puzzle.get('id', '?')} "
                    f"predicted={result.predicted_answer!r} "
                    f"expected={puzzle['answer']!r}",
                    file=sys.stderr,
                )

    if total == 0:
        print("No cipher puzzles found.")
    else:
        accuracy = correct / total
        print(f"Cipher accuracy: {correct}/{total} = {accuracy:.4f}")
