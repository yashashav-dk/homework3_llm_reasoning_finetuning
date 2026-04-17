from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SolverResult:
    puzzle_id: str
    category: str
    predicted_answer: str
    is_correct: bool
    solve_method: str = ""
    confidence: float = 1.0


class BaseSolver(ABC):
    category: str  # subclass must set this

    @abstractmethod
    def solve(self, puzzle: dict) -> SolverResult:
        """Given puzzle dict with 'id', 'prompt', 'answer', return SolverResult."""
        ...

    def verify(self, puzzle: dict) -> bool:
        """Check if solver's answer matches ground truth using competition metric."""
        from src.metrics.competition import verify
        result = self.solve(puzzle)
        return verify(puzzle["answer"], result.predicted_answer)
