"""Chain-of-thought trace generator for substitution cipher puzzles."""

from __future__ import annotations

import re

from src.solvers.base import SolverResult
from src.solvers.cipher import (
    _VOCAB_BY_LEN,
    _build_char_mapping,
    _parse_examples,
    _parse_target,
    _vocab_complete,
)
from src.trace_generators.base import BaseTraceGenerator, CoTTrace


class CipherTraceGenerator(BaseTraceGenerator):
    category = "cipher"

    def generate_trace(self, puzzle: dict, solver_result: SolverResult) -> CoTTrace:
        puzzle_id = puzzle.get("id", "")
        prompt: str = puzzle.get("prompt", "")

        # ------------------------------------------------------------------ #
        # Re-derive all intermediate state so the trace is self-consistent.   #
        # ------------------------------------------------------------------ #
        examples = _parse_examples(prompt)
        target_enc = _parse_target(prompt)
        mapping = _build_char_mapping(examples)

        lines: list[str] = []

        # ------------------------------------------------------------------ #
        # Step 1 – announce intent                                            #
        # ------------------------------------------------------------------ #
        lines.append(
            "Let me analyze the encryption examples to build a character mapping.\n"
        )

        # ------------------------------------------------------------------ #
        # Step 2 – walk through each example pair                             #
        # ------------------------------------------------------------------ #
        if examples:
            lines.append("=== Extracting mappings from example pairs ===\n")
            running: dict[str, str] = {}
            for i, (enc_word, dec_word) in enumerate(examples, start=1):
                lines.append(f"Example {i}: '{enc_word}' → '{dec_word}'")
                new_mappings: list[str] = []
                for enc_ch, dec_ch in zip(enc_word, dec_word):
                    if enc_ch not in running:
                        running[enc_ch] = dec_ch
                        new_mappings.append(f"  {enc_ch!r} → {dec_ch!r}")
                    else:
                        # Already seen — confirm consistency
                        if running[enc_ch] == dec_ch:
                            lines.append(
                                f"  {enc_ch!r} → {dec_ch!r}  (confirms existing mapping)"
                            )
                        else:
                            lines.append(
                                f"  WARNING: conflict for {enc_ch!r}: "
                                f"previously mapped to {running[enc_ch]!r}, "
                                f"now {dec_ch!r} — keeping first."
                            )
                if new_mappings:
                    lines.extend(new_mappings)
                lines.append("")
        else:
            lines.append(
                "(No structured example pairs found in prompt — "
                "will rely on vocabulary matching.)\n"
            )
            running = {}

        # ------------------------------------------------------------------ #
        # Step 3 – print the full mapping table built so far                  #
        # ------------------------------------------------------------------ #
        lines.append("=== Full character mapping table ===")
        if mapping:
            # Sort by encrypted character for readability
            for enc_ch in sorted(mapping):
                lines.append(f"  '{enc_ch}' → '{mapping[enc_ch]}'")
        else:
            lines.append("  (empty — no examples parsed)")
        lines.append("")

        # ------------------------------------------------------------------ #
        # Step 4 – decrypt target character by character / word by word       #
        # ------------------------------------------------------------------ #
        lines.append(f"=== Decrypting target: '{target_enc}' ===\n")

        working_mapping = dict(mapping)

        # Tokenise the same way _decrypt_text does
        tokens = re.split(r"(\W+)", target_enc)
        decrypted_tokens: list[str] = []

        for token in tokens:
            if not token:
                continue

            if not token.isalpha():
                # Non-alpha: map char by char
                part = ""
                for ch in token:
                    mapped = working_mapping.get(ch, ch)
                    if ch == mapped:
                        # space / punctuation — keep as-is
                        pass
                    else:
                        lines.append(
                            f"  Non-alpha '{ch}' → '{mapped}' (via mapping)"
                        )
                    part += mapped
                decrypted_tokens.append(part)
                continue

            enc_word = token.lower()
            lines.append(f"Word: '{enc_word}'")

            # Character-by-character lookup
            char_results: list[str] = []
            has_unknown = False
            for enc_ch in enc_word:
                if enc_ch in working_mapping:
                    dec_ch = working_mapping[enc_ch]
                    lines.append(
                        f"  '{enc_ch}' → '{dec_ch}'  (from mapping table)"
                    )
                    char_results.append(dec_ch)
                else:
                    lines.append(
                        f"  '{enc_ch}' → ?  (not in mapping — unknown)"
                    )
                    char_results.append("?")
                    has_unknown = True

            partial = "".join(char_results)

            if has_unknown:
                lines.append(
                    f"  Partial decryption: '{partial}' — "
                    f"attempting vocabulary matching ..."
                )
                candidates_for_len = _VOCAB_BY_LEN[len(enc_word)]
                lines.append(
                    f"  Vocabulary candidates of length {len(enc_word)}: "
                    + (
                        ", ".join(candidates_for_len[:10])
                        + (" ..." if len(candidates_for_len) > 10 else "")
                        if candidates_for_len
                        else "(none)"
                    )
                )
                completed = _vocab_complete(partial, enc_word, working_mapping)
                if completed:
                    lines.append(
                        f"  Matched vocabulary word: '{completed}'"
                    )
                    # Update mapping
                    for enc_ch, dec_ch in zip(enc_word, completed):
                        if enc_ch not in working_mapping:
                            lines.append(
                                f"  Extended mapping: '{enc_ch}' → '{dec_ch}' "
                                f"(inferred from vocab match)"
                            )
                            working_mapping[enc_ch] = dec_ch
                    partial = completed
                else:
                    lines.append(
                        "  No unique vocabulary match — keeping '?' for unknown chars."
                    )

            # Restore original casing
            out_word = ""
            for orig_ch, dec_ch in zip(token, partial):
                out_word += dec_ch.upper() if orig_ch.isupper() else dec_ch
            lines.append(f"  → '{out_word}'\n")
            decrypted_tokens.append(out_word)

        decrypted_text = "".join(decrypted_tokens)

        # ------------------------------------------------------------------ #
        # Step 5 – final answer                                               #
        # ------------------------------------------------------------------ #
        lines.append("=== Result ===")
        lines.append(f"Encrypted:  '{target_enc}'")
        lines.append(f"Decrypted:  '{decrypted_text}'")
        lines.append("")
        lines.append(f"The answer is {decrypted_text}")

        thinking_text = "\n".join(lines)

        return CoTTrace(
            puzzle_id=puzzle_id,
            category=self.category,
            thinking_text=thinking_text,
            final_answer=decrypted_text,
            token_count=self.count_tokens(thinking_text),
            is_verified=solver_result.is_correct,
        )


# ---------------------------------------------------------------------------
# CLI: demo on a hand-crafted cipher puzzle
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.solvers.cipher import CipherSolver

    demo_puzzle = {
        "id": "cipher_demo_001",
        "category": "cipher",
        "prompt": (
            "'dpodfqu' encrypts to 'concept'\n"
            "'bmjdf' encrypts to 'alice'\n"
            "'xpoefsmboe' encrypts to 'wonderland'\n"
            "What does 'xpoefsgmboe' decrypt to?"
        ),
        "answer": "wondergland",
    }

    solver = CipherSolver()
    result = solver.solve(demo_puzzle)
    print("Solver result:", result)
    print()

    generator = CipherTraceGenerator()
    trace = generator.generate_trace(demo_puzzle, result)
    print("=== CoT Trace ===")
    print(trace.thinking_text)
    print()
    print(f"Final answer : {trace.final_answer!r}")
    print(f"Token count  : {trace.token_count}")
    print(f"Verified     : {trace.is_verified}")
