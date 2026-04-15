# Contract: Experiment Interface

**Revision**: 2 (corrected for 30B model, adapter submission)

## Experiment Configuration (YAML)

Every experiment MUST have a YAML config file committed before
the run begins.

```yaml
# Required fields
model_name: string          # MUST be nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
seed: int                   # Random seed (required, no default)
quantization: string        # none | fp8 | nf4

# Training fields (required for fine-tuning experiments)
lora_rank: int              # LoRA rank (MUST be <= 32)
lora_alpha: int             # LoRA alpha scaling
lora_target_modules: list   # Target modules (e.g., [q_proj, k_proj, v_proj, o_proj])
learning_rate: float        # Peak learning rate
batch_size: int             # Per-device batch size
gradient_accumulation: int  # Gradient accumulation steps
max_seq_length: int         # Maximum sequence length
num_epochs: int             # Number of training epochs
warmup_ratio: float         # Learning rate warmup ratio

# Evaluation fields (MUST match competition — do not change)
max_new_tokens: 7680        # Competition-fixed
temperature: 0.0            # Competition-fixed (greedy)
top_p: 1.0                  # Competition-fixed
max_model_len: 8192         # Competition-fixed

# Metadata
prompt_template: string     # Path to prompt template YAML
dataset_id: string          # Dataset identifier (null for eval-only)
treatment_variable: string  # What changed vs. control
control_run_id: string      # Reference control experiment
```

## Experiment Log (CSV)

Append-only CSV at `experiments/experiment_log.csv`.

```csv
run_id,config_hash,timestamp,hypothesis,treatment_variable,
control_run_id,seed,model_id,adapter_path,prompt_template_id,
dataset_id,status,overall_accuracy,math_accuracy,code_accuracy,
logic_accuracy,boxed_rate,truncation_rate,total_time_seconds,
peak_vram_gb,delta_vs_control,decision,notes
```

## Prompt Template (YAML)

```yaml
template_id: string
system_prompt: |
  Multi-line system prompt. MUST instruct model to place
  final answer in \boxed{} format.
few_shot_examples:
  - domain: math | code | logic
    input: "Problem statement"
    output: "Reasoning chain ending with \\boxed{answer}"
format_instructions: |
  Place your final answer in \boxed{} format.
```

## LoRA Adapter Submission (ZIP)

`submission.zip` MUST contain:

```text
submission.zip
├── adapter_config.json     # PEFT adapter config (rank <= 32)
└── adapter_model.safetensors  # Adapter weights
```

`adapter_config.json` MUST specify:
- `base_model_name_or_path`: the 30B model ID
- `r`: <= 32
- `task_type`: "CAUSAL_LM"

## Competition Metric

Accuracy = (correct predictions) / (total predictions)

A prediction is correct if:
1. Extract answer from `\boxed{}` (priority) or fallback heuristics
2. Compare to ground truth:
   - Exact string match, OR
   - Numerical match within relative tolerance 1e-4
