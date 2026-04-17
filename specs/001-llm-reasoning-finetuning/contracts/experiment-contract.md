# Contract: Experiment Interface

**Revision**: 3 (solver-first architecture)

## Solver Interface

Every solver MUST implement:

```python
class BaseSolver:
    def solve(self, puzzle: dict) -> SolverResult:
        """Given a puzzle dict with 'id', 'prompt', 'answer',
        return a SolverResult with predicted_answer."""
        ...

    def verify(self, puzzle: dict) -> bool:
        """Check if solver's answer matches ground truth
        using competition verify() function."""
        ...
```

## Trace Generator Interface

Every trace generator MUST produce:

```python
class BaseTraceGenerator:
    def generate_trace(self, puzzle: dict,
                       solver_result: SolverResult) -> CoTTrace:
        """Generate a step-by-step reasoning trace that:
        1. Fits within 7680 tokens
        2. Uses <think>...</think> format
        3. Ends with \\boxed{answer}
        4. Is deterministic (same puzzle = same trace)
        """
        ...
```

## SFT Data Format (JSONL)

```json
{
    "messages": [
        {
            "role": "user",
            "content": "{puzzle_prompt}\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"
        },
        {
            "role": "assistant",
            "content": "<think>\n{step_by_step_reasoning}\n</think>\n\n\\boxed{final_answer}"
        }
    ],
    "puzzle_id": "abc123",
    "category": "bit_manipulation",
    "token_count": 2345
}
```

## LoRA Adapter Submission (ZIP)

```text
submission.zip
├── adapter_config.json
└── adapter_model.safetensors
```

`adapter_config.json` MUST specify:
- `r`: <= 32
- `target_modules`: matching `r".*\.(in_proj|out_proj|up_proj|down_proj)$"`
- `task_type`: "CAUSAL_LM"

## Competition Metric

```python
# Answer extraction priority:
# 1. \boxed{...} (last non-empty match)
# 2. "The final answer is:" patterns
# 3. Last numeric value
# 4. Last non-empty line

# Answer verification:
# Binary [01]+: strict case-insensitive string match
# Numeric: math.isclose(rel_tol=1e-2, abs_tol=1e-5)
# Other: case-insensitive string comparison
```
