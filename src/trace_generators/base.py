from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.solvers.base import SolverResult


@dataclass
class CoTTrace:
    puzzle_id: str
    category: str
    thinking_text: str
    final_answer: str
    token_count: int = 0
    is_verified: bool = False


class BaseTraceGenerator(ABC):
    category: str  # subclass must set this
    max_tokens: int = 7680
    _tokenizer = None
    _tokenizer_available: bool | None = None

    @classmethod
    def _get_tokenizer(cls):
        if cls._tokenizer_available is None:
            try:
                from transformers import AutoTokenizer

                cls._tokenizer = AutoTokenizer.from_pretrained(
                    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                    trust_remote_code=True,
                )
                cls._tokenizer_available = True
            except (ImportError, OSError):
                cls._tokenizer_available = False
        return cls._tokenizer

    @abstractmethod
    def generate_trace(self, puzzle: dict, solver_result: SolverResult) -> CoTTrace:
        """Generate a step-by-step reasoning trace.

        Must:
        1. Fit within max_tokens
        2. Be deterministic (same puzzle = same trace)
        3. Produce verified correct answer
        """
        ...

    def count_tokens(self, text: str) -> int:
        """Count tokens using the actual model tokenizer, with char-estimate fallback."""
        tok = self._get_tokenizer()
        if tok is not None:
            return len(tok.encode(text, add_special_tokens=False))
        # Fallback: ~1 token per 4 chars (conservative overcount is safer)
        return len(text) // 4
