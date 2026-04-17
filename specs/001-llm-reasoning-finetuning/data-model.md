# Data Model: LLM Reasoning Fine-Tuning Pipeline

**Branch**: `001-llm-reasoning-finetuning` | **Date**: 2026-04-16
**Revision**: 3 (rebuilt for solver-first architecture)

## Entities

### Puzzle

A single puzzle from train.csv with its category classification.

| Field | Type | Description |
|-------|------|-------------|
| id | string | Unique puzzle identifier (from train.csv) |
| prompt | string | Full puzzle text with examples |
| answer | string | Ground truth answer |
| category | enum | numeral / gravity / unit_conversion / cipher / bit_manipulation / equation_numeric_deduce / equation_numeric_guess / cryptarithm_deduce / cryptarithm_guess |
| split | enum | train / validation |

### SolverResult

Output from running a category-specific solver on a puzzle.

| Field | Type | Description |
|-------|------|-------------|
| puzzle_id | string | Reference to puzzle |
| category | enum | Puzzle category |
| predicted_answer | string | Solver's answer |
| is_correct | bool | verify(answer, predicted_answer) |
| solve_method | string | Which algorithm/gate/operator matched |
| confidence | float | 1.0 for deterministic, <1.0 for heuristic |

### CoTTrace

A generated chain-of-thought reasoning trace for training.

| Field | Type | Description |
|-------|------|-------------|
| puzzle_id | string | Reference to puzzle |
| category | enum | Puzzle category |
| thinking_text | string | Content inside `<think>...</think>` |
| final_answer | string | Content inside `\boxed{}` |
| token_count | int | Total tokens (must be < 7680) |
| is_verified | bool | Solver confirms trace produces correct answer |

### SFTExample

A formatted training example ready for QLoRA fine-tuning.

| Field | Type | Description |
|-------|------|-------------|
| puzzle_id | string | Reference to puzzle |
| messages | list | Chat-format messages (user + assistant) |
| total_tokens | int | Must be < 7680 |

User message format:
```
{puzzle.prompt}
Please put your final answer inside `\boxed{}`. For example: `\boxed{your answer}`
```

Assistant message format:
```
<think>
{trace.thinking_text}
</think>

\boxed{{trace.final_answer}}
```

### Experiment

Unchanged from previous revision — tracks each training/eval run.

| Field | Type | Description |
|-------|------|-------------|
| run_id | string | Unique identifier |
| config_hash | string | SHA256 of config file |
| timestamp | datetime | Run start time |
| hypothesis | string | What this experiment tests |
| treatment_variable | string | Single variable changed |
| control_run_id | string | Reference to baseline |
| seed | int | Random seed |
| categories_included | list | Which puzzle categories in training data |
| status | enum | pending / running / completed / failed |

### ExperimentResult

| Field | Type | Description |
|-------|------|-------------|
| run_id | string | Reference to experiment |
| overall_accuracy | float | Aggregate accuracy |
| per_category_accuracy | dict | {category: accuracy} for all 7+ |
| min_logprob | float | Minimum logprob across all traces |
| total_time_seconds | float | Wall-clock time |
| peak_vram_gb | float | Peak GPU memory |
| decision | enum | adopt / revert / investigate |

### LoRAAdapter

| Field | Type | Description |
|-------|------|-------------|
| adapter_id | string | Unique identifier |
| run_id | string | Experiment that produced it |
| rank | int | LoRA rank (must be <= 32) |
| target_modules | string | Regex pattern for target modules |
| local_accuracy | float | Accuracy on local validation split |
| per_category_accuracy | dict | Per-category breakdown |

## Relationships

```text
Puzzle 1──1 SolverResult (per solver run)
Puzzle 1──1 CoTTrace (per trace generation)
CoTTrace 1──1 SFTExample (formatting step)
Experiment 1──N SFTExample (training data)
Experiment 1──1 ExperimentResult
Experiment 1──N LoRAAdapter
```

## Storage Format

```text
data/
├── train.csv                    # Competition data (9500 puzzles)
├── puzzles_classified.jsonl     # Puzzles with category labels
├── splits/
│   ├── train_ids.json           # 90% puzzle IDs for training
│   └── val_ids.json             # 10% puzzle IDs for evaluation
├── solver_results/
│   ├── numeral_results.jsonl
│   ├── gravity_results.jsonl
│   └── ...
├── traces/
│   ├── numeral_traces.jsonl
│   ├── gravity_traces.jsonl
│   └── ...
└── sft/
    └── train_sft.jsonl          # Formatted SFT examples

experiments/
├── experiment_log.csv
├── configs/
└── results/

checkpoints/
├── exp-010-easy/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── best/
```
