# Data Model: LLM Reasoning Fine-Tuning Pipeline

**Branch**: `001-llm-reasoning-finetuning` | **Date**: 2026-04-14
**Revision**: 2 (corrected for 30B model, adapter submission)

## Entities

### Experiment

Represents a single experimental run with tracked configuration
and results.

| Field | Type | Description |
|-------|------|-------------|
| run_id | string | Unique identifier (e.g., `exp-001-baseline`) |
| config_hash | string | SHA256 of the config file |
| timestamp | datetime | Run start time (ISO 8601) |
| hypothesis | string | What this experiment tests |
| treatment_variable | string | Single variable changed (ablation) |
| control_run_id | string | Reference to baseline/control run |
| seed | int | Random seed (PyTorch + NumPy + Python) |
| model_id | string | Always `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| adapter_path | string | Path to LoRA adapter (null if base model) |
| prompt_template_id | string | Reference to prompt template used |
| dataset_id | string | Reference to training dataset version |
| status | enum | pending / running / completed / failed |

### ExperimentConfig

YAML configuration file committed alongside each run.

| Field | Type | Description |
|-------|------|-------------|
| model_name | string | HuggingFace model ID (30B) |
| seed | int | Random seed (required, no default) |
| quantization | string | none / fp8 / nf4 (for QLoRA) |

**Training-only fields**:

| Field | Type | Description |
|-------|------|-------------|
| lora_rank | int | LoRA rank (max 32, competition limit) |
| lora_alpha | int | LoRA alpha scaling |
| lora_target_modules | list | Target modules for LoRA |
| learning_rate | float | Peak learning rate |
| batch_size | int | Per-device batch size |
| gradient_accumulation | int | Gradient accumulation steps |
| max_seq_length | int | Maximum sequence length |
| num_epochs | int | Training epochs |
| warmup_ratio | float | LR warmup ratio |

**Evaluation fields (must match competition)**:

| Field | Type | Description |
|-------|------|-------------|
| max_new_tokens | int | 7680 (competition-fixed) |
| temperature | float | 0.0 (competition-fixed) |
| top_p | float | 1.0 (competition-fixed) |
| max_model_len | int | 8192 (competition-fixed) |

**Metadata**:

| Field | Type | Description |
|-------|------|-------------|
| prompt_template | string | Path to prompt template YAML |
| dataset_id | string | Dataset identifier (null for eval-only) |
| treatment_variable | string | What changed vs. control |
| control_run_id | string | Reference control experiment |

### ExperimentResult

Metrics recorded after each run.

| Field | Type | Description |
|-------|------|-------------|
| run_id | string | Reference to experiment |
| overall_accuracy | float | Aggregate accuracy (competition metric) |
| math_accuracy | float | Accuracy on math domain |
| code_accuracy | float | Accuracy on code domain |
| logic_accuracy | float | Accuracy on logic domain |
| boxed_rate | float | % of responses with valid \boxed{} |
| truncation_rate | float | % of responses hitting token limit |
| total_time_seconds | float | Wall-clock time |
| peak_vram_gb | float | Peak GPU memory usage |
| num_samples_evaluated | int | Number of eval samples |
| delta_vs_control | float | Accuracy change vs. control run |
| decision | enum | adopt / revert / investigate |
| notes | string | Free-text observations |

### PromptTemplate

System prompt and few-shot example configuration.

| Field | Type | Description |
|-------|------|-------------|
| template_id | string | Unique identifier |
| system_prompt | string | System message (must instruct \boxed{}) |
| few_shot_examples | list | Domain-tagged example pairs |
| format_instructions | string | \boxed{} format guidance |

### TrainingDataset

Curated dataset with provenance tracking.

| Field | Type | Description |
|-------|------|-------------|
| dataset_id | string | Unique identifier + version |
| source | string | `nvidia/OpenMathReasoning` |
| solution_type | string | cot / tir / genselect |
| license | string | CC-BY-4.0 |
| num_samples | int | Number of training samples |
| filtering_method | string | How data was filtered |
| max_token_length | int | Longest solution (tokens) |
| boxed_format_rate | float | % with valid \boxed{} answers |
| domain_distribution | dict | Counts by domain category |

### LoRAAdapter

Fine-tuned adapter weights for submission.

| Field | Type | Description |
|-------|------|-------------|
| adapter_id | string | Unique identifier |
| run_id | string | Experiment that produced this |
| rank | int | LoRA rank (must be <= 32) |
| alpha | int | LoRA alpha |
| target_modules | list | Which model layers adapted |
| base_model | string | Must be Nemotron-3-Nano-30B |
| adapter_config_path | string | Path to adapter_config.json |
| weights_path | string | Path to adapter weights |
| local_accuracy | float | Accuracy on local validation split |

### Submission

Kaggle submission artifact.

| Field | Type | Description |
|-------|------|-------------|
| submission_id | string | Unique identifier |
| adapter_id | string | LoRA adapter used |
| local_score | float | Score on local validation split |
| public_score | float | Kaggle public leaderboard (null pre-submit) |
| zip_path | string | Path to submission.zip |
| submitted_at | datetime | Submission timestamp |

## Relationships

```text
Experiment 1──1 ExperimentConfig
Experiment 1──1 ExperimentResult
Experiment N──1 PromptTemplate
Experiment N──1 TrainingDataset (nullable, eval-only runs)
Experiment 1──N LoRAAdapter (training runs produce adapters)
LoRAAdapter 1──N Submission
Experiment N──1 Experiment (control_run_id → parent baseline)
```

## Storage Format

All entities stored as flat files:

```text
experiments/
├── experiment_log.csv       # One row per experiment (append-only)
├── configs/
│   ├── exp-001-baseline.yaml
│   ├── exp-002-cot-prompt.yaml
│   └── ...
└── results/
    ├── exp-001-baseline.json
    └── ...

prompts/
├── baseline.yaml
├── cot-math.yaml
└── ...

checkpoints/
├── exp-010-qlora-r16/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── exp-011-qlora-r32/
    ├── adapter_config.json
    └── adapter_model.safetensors

submissions/
├── submission-001.zip
└── ...
```

## Validation Rules

- `run_id` MUST be unique across all experiments
- `seed` MUST be explicitly set (no null/random default)
- `config_hash` MUST match the committed config file
- `treatment_variable` MUST name exactly one changed variable
  (or "compound: X, Y" with justification)
- `lora_rank` MUST be <= 32 (competition enforced)
- `model_id` MUST be `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- `decision` MUST be set before starting the next experiment
- Evaluation params (temperature, top_p, max_tokens) MUST match
  competition values exactly
- All datasets MUST have `license` = CC-BY-4.0 or equivalent
  permissive license
