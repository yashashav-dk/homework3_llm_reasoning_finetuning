"""Solver for substitution cipher puzzles (Alice in Wonderland themed)."""

from __future__ import annotations

import re
from collections import defaultdict

from src.solvers.base import BaseSolver, SolverResult

# ---------------------------------------------------------------------------
# Alice in Wonderland vocabulary (~90 common words)
# ---------------------------------------------------------------------------

WONDERLAND_VOCAB: list[str] = [
    # Characters
    "alice", "rabbit", "queen", "king", "hatter", "cheshire", "cat", "mouse",
    "duchess", "cook", "caterpillar", "dormouse", "march", "hare", "dodo",
    "mock", "turtle", "gryphon", "lobster", "pigeon", "duchess", "knave",
    # Places / things
    "wonderland", "garden", "hole", "door", "table", "chair", "cup", "tea",
    "party", "house", "pool", "court", "trial", "jury", "card", "cards",
    "heart", "hearts", "diamond", "spade", "club", "bottle", "cake", "key",
    "glass", "mirror", "mushroom", "hookah", "flamingo", "hedgehog", "croquet",
    # Common Alice-story words
    "curiouser", "curious", "shrink", "grow", "small", "large", "size",
    "drink", "eat", "fall", "falling", "down", "up", "again", "away",
    "moment", "suddenly", "thought", "said", "began", "felt", "found",
    "quite", "very", "little", "great", "white", "red", "black", "golden",
    "time", "way", "head", "hand", "feet", "foot", "eye", "eyes", "voice",
    "nothing", "something", "everything", "anything", "somebody", "nobody",
    "wonder", "dream", "sleep", "wake", "run", "jump", "stop", "come",
    "went", "into", "upon", "under", "about", "after", "before", "back",
    "round", "long", "looked", "began", "right", "left", "next", "just",
    "read", "written", "write", "know", "think", "see", "hear", "feel",
    "word", "letter", "name", "rule", "must", "could", "would", "should",
    "mad", "crazy", "strange", "odd", "silly", "nonsense", "sentence",
    "pepper", "soup", "tarts", "bread", "butter", "jam",
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

    Expected prompt structure (flexible):
      - Lines that show  "<encrypted>" → "<decrypted>"  or
        "<encrypted>" encrypts to "<decrypted>"  or
        "<encrypted>" = "<decrypted>"  etc.
    Also handles the question line:
      What does '<target>' decrypt to?
    Returns only example pairs (both sides non-empty).
    """
    examples: list[tuple[str, str]] = []

    # A token is either a single-quoted word, a double-quoted word, or a bare
    # run of word-characters (no embedded spaces in substitution cipher words).
    _TOK = r"""(?:'(\w[\w\s]*?)'|"(\w[\w\s]*?)"|(\w+))"""

    def _extract_tok(m_groups: tuple) -> str:
        """Return the first non-None group from a _TOK match triplet."""
        for g in m_groups:
            if g is not None:
                return g.strip()
        return ""

    seen: set[tuple[str, str]] = set()

    def _add(enc: str, dec: str) -> None:
        enc, dec = enc.strip().lower(), dec.strip().lower()
        if enc and dec and enc != dec:
            key = (enc, dec)
            if key not in seen:
                seen.add(key)
                examples.append((enc, dec))

    # Build composite patterns using the robust _TOK sub-pattern.
    # Each pattern captures two groups of three (from two _TOK occurrences).
    # Total groups per match: 6.  enc = groups 0-2, dec = groups 3-5.

    # 1. "X" encrypts to "Y"
    enc_pattern = re.compile(
        _TOK + r"\s+encrypts?\s+to\s+" + _TOK, re.IGNORECASE
    )
    # 2. "X" decrypts to "Y"  (enc shown first, plain second)
    dec_pattern = re.compile(
        _TOK + r"\s+decrypts?\s+to\s+" + _TOK, re.IGNORECASE
    )
    # 3. Arrow forms:  "X" -> "Y"  /  X → Y
    arrow_pattern = re.compile(
        _TOK + r"\s*(?:->|→|=>)\s*" + _TOK, re.IGNORECASE
    )
    # 4. Equals form: 'X' = 'Y'  (both must be quoted to avoid matching math)
    equals_pattern = re.compile(
        r"""(?:'(\w+)'|"(\w+)")\s*=\s*(?:'(\w+)'|"(\w+)")""", re.IGNORECASE
    )
    # 5. encrypted: "X", decrypted: "Y"
    field_pattern = re.compile(
        r"encrypted[:\s]+" + _TOK + r"\s*[,;]\s*decrypted[:\s]+" + _TOK,
        re.IGNORECASE,
    )

    for pattern in (enc_pattern, dec_pattern, arrow_pattern, field_pattern):
        for m in pattern.finditer(prompt):
            groups = m.groups()
            enc = _extract_tok(groups[:3])
            dec = _extract_tok(groups[3:6])
            _add(enc, dec)

    # equals_pattern: 4 groups total, enc = groups[0:2], dec = groups[2:4]
    for m in equals_pattern.finditer(prompt):
        groups = m.groups()
        enc = _extract_tok(groups[:2] + ("",))   # pad to 3 for _extract_tok
        dec = _extract_tok(groups[2:4] + ("",))
        _add(enc, dec)

    # Fallback: scan each line for exactly two quoted words → (enc, dec)
    if not examples:
        for line in prompt.splitlines():
            # Skip lines that look like the question
            if re.search(r"\bdecrypt\b", line, re.IGNORECASE) and "?" in line:
                continue
            quoted = re.findall(r"['\"](\w+)['\"]", line)
            if len(quoted) == 2:
                _add(quoted[0], quoted[1])

    return examples


def _parse_target(prompt: str) -> str:
    """Extract the target encrypted string to decrypt."""
    # "What does 'X' decrypt to?"  /  "Decrypt 'X'"  / "What is 'X'?"
    patterns = [
        re.compile(r"what\s+does\s+['\"]([^'\"]+)['\"]?\s+decrypt", re.IGNORECASE),
        re.compile(r"decrypt\s+['\"]([^'\"]+)['\"]", re.IGNORECASE),
        re.compile(r"what\s+is\s+['\"]([^'\"]+)['\"]", re.IGNORECASE),
        re.compile(r"['\"]([^'\"]+)['\"]?\s*decrypts?\s+to\s*\?", re.IGNORECASE),
        re.compile(r"['\"]([^'\"]+)['\"]?\s*=\s*\?", re.IGNORECASE),
    ]
    for pat in patterns:
        m = pat.search(prompt)
        if m:
            return m.group(1).strip()

    # Fallback: last quoted token in prompt
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", prompt)
    if quoted:
        return quoted[-1].strip()

    # Last non-empty line after the examples block
    lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        # Strip trailing punctuation
        return last.rstrip("?!.:;")

    return ""


# ---------------------------------------------------------------------------
# Mapping builder
# ---------------------------------------------------------------------------

def _build_char_mapping(examples: list[tuple[str, str]]) -> dict[str, str]:
    """Build encrypted_char → decrypted_char from all example pairs."""
    mapping: dict[str, str] = {}
    for enc_word, dec_word in examples:
        if len(enc_word) != len(dec_word):
            # Skip mismatched-length pairs (shouldn't happen in well-formed puzzles)
            continue
        for enc_ch, dec_ch in zip(enc_word, dec_word):
            if enc_ch not in mapping:
                mapping[enc_ch] = dec_ch
            # If conflict, keep first assignment (majority could be used, but
            # consistent ciphers never conflict).
    return mapping


# ---------------------------------------------------------------------------
# Vocabulary-based gap filling
# ---------------------------------------------------------------------------

def _apply_mapping_to_word(enc_word: str, mapping: dict[str, str]) -> str:
    """Apply known mapping to a single word; return partial result with '?' for unknowns."""
    return "".join(mapping.get(ch, "?") for ch in enc_word)


def _vocab_complete(partial: str, enc_word: str, mapping: dict[str, str]) -> str | None:
    """Try to complete a partially-mapped word using WONDERLAND_VOCAB.

    Returns the completed plain word if exactly one candidate matches, else None.
    Candidate must:
    - Match length
    - Have all known positions match
    - Be consistent with any already-established mappings (no conflicts)
    """
    n = len(partial)
    candidates = []
    for vocab_word in _VOCAB_BY_LEN[n]:
        ok = True
        for i, ch in enumerate(partial):
            if ch != "?" and ch != vocab_word[i]:
                ok = False
                break
        if not ok:
            continue
        # Check that the mapping extensions introduced by this candidate
        # don't conflict with the existing mapping.
        consistent = True
        for enc_ch, plain_ch in zip(enc_word, vocab_word):
            if enc_ch in mapping and mapping[enc_ch] != plain_ch:
                consistent = False
                break
        if consistent:
            candidates.append(vocab_word)

    if len(candidates) == 1:
        return candidates[0]
    return None


# ---------------------------------------------------------------------------
# Core decryption
# ---------------------------------------------------------------------------

def _decrypt_text(
    enc_text: str,
    mapping: dict[str, str],
    *,
    use_vocab: bool = True,
) -> tuple[str, dict[str, str]]:
    """Decrypt enc_text using mapping.

    For each word, attempt vocabulary completion for any unmapped characters.
    Returns (decrypted_text, updated_mapping).
    """
    working_mapping = dict(mapping)

    # Tokenise preserving spaces and punctuation structure
    # Split into tokens: alpha-runs and non-alpha runs
    tokens = re.split(r"(\W+)", enc_text)
    result_parts: list[str] = []

    for token in tokens:
        if not token:
            continue
        if not token.isalpha():
            # Non-alpha: map each character; spaces/punctuation typically identity
            part = ""
            for ch in token:
                part += working_mapping.get(ch, ch)
            result_parts.append(part)
            continue

        # Alpha word token
        enc_word = token.lower()
        partial = _apply_mapping_to_word(enc_word, working_mapping)

        if "?" in partial and use_vocab:
            completed = _vocab_complete(partial, enc_word, working_mapping)
            if completed:
                # Update mapping with new character correspondences
                for enc_ch, dec_ch in zip(enc_word, completed):
                    if enc_ch not in working_mapping:
                        working_mapping[enc_ch] = dec_ch
                partial = completed

        # Preserve original case pattern
        out_word = ""
        for orig_ch, dec_ch in zip(token, partial):
            if orig_ch.isupper():
                out_word += dec_ch.upper()
            else:
                out_word += dec_ch
        result_parts.append(out_word)

    return "".join(result_parts), working_mapping


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
        predicted, _ = _decrypt_text(target_enc, mapping, use_vocab=True)

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
