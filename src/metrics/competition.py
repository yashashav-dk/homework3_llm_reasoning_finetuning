"""Competition metric for the NVIDIA Nemotron reasoning benchmark."""

import math
import re


def extract_final_answer(text: str) -> str:
    """Extract the final answer from model output.

    Priority 1: Last non-empty \\boxed{...} match (handles nested braces).
    Priority 2: Fallback natural-language patterns.
    Priority 3: Last numeric value in text.
    Priority 4: Last non-empty line.
    Returns 'NOT_FOUND' if text is None.
    """
    if text is None:
        return "NOT_FOUND"

    # Priority 1: \boxed{...} with nested-brace handling
    boxed_results = []
    search_start = 0
    while True:
        match_start = text.find(r"\boxed{", search_start)
        if match_start == -1:
            break
        # Start counting after the opening brace of \boxed{
        depth = 0
        content_start = match_start + len(r"\boxed{")
        i = content_start
        found_close = False
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                if depth == 0:
                    # This closes the \boxed{
                    content = text[content_start:i]
                    if content.strip():
                        boxed_results.append(content)
                    found_close = True
                    search_start = i + 1
                    break
                else:
                    depth -= 1
            i += 1
        if not found_close:
            break

    if boxed_results:
        return boxed_results[-1]

    # Priority 2: Natural-language fallback patterns
    fallback_patterns = [
        r"[Tt]he\s+final\s+answer\s+is[:\s]+(.+?)(?:\n|$)",
        r"[Ff]inal\s+answer\s+is[:\s]+(.+?)(?:\n|$)",
        r"[Tt]he\s+answer\s+is[:\s]+(.+?)(?:\n|$)",
        r"[Aa]nswer[:\s]+(.+?)(?:\n|$)",
        r"[Tt]herefore[,\s]+(.+?)(?:\n|$)",
    ]
    for pattern in fallback_patterns:
        match = re.search(pattern, text)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate

    # Priority 3: Last numeric value
    numeric_matches = list(re.finditer(r"-?\d+(?:\.\d+)?", text))
    if numeric_matches:
        return numeric_matches[-1].group(0)

    # Priority 4: Last non-empty line
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped

    return "NOT_FOUND"


def verify(stored_answer: str, predicted: str) -> bool:
    """Compare predicted answer against ground truth.

    Binary strings (regex ^[01]+$): strict case-insensitive string match.
    Numeric (both parseable as float): math.isclose(rel_tol=1e-2, abs_tol=1e-5).
    Otherwise: case-insensitive stripped string comparison.
    """
    stored = stored_answer.strip()
    pred = predicted.strip()

    binary_pattern = re.compile(r"^[01]+$")

    # Binary strings: strict case-insensitive match (no leading-zero tolerance)
    if binary_pattern.match(stored) and binary_pattern.match(pred):
        return stored.lower() == pred.lower()

    # Numeric comparison
    try:
        stored_float = float(stored)
        pred_float = float(pred)
        return math.isclose(stored_float, pred_float, rel_tol=1e-2, abs_tol=1e-5)
    except ValueError:
        pass

    # Fallback: case-insensitive stripped string comparison
    return stored.lower() == pred.lower()
