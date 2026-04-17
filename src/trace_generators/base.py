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
        """Estimate token count. Override with tokenizer-based counting."""
        # Rough estimate: 1 token per 4 chars
        return len(text) // 4
