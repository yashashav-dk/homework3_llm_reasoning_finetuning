# Plan: Unsloth QLoRA Training on Colab A100-80GB

**Date**: 2026-04-22
**Status**: Ready to execute
**Prerequisite**: Colab Pro with A100 High-RAM GPU, Kaggle API credentials in Colab Secrets

## Context

Previous attempts to train with plain transformers + PEFT + bitsandbytes failed
because:
1. `mamba_ssm` package won't build on Colab (PyTorch 2.10 + CUDA 12.8)
2. Patching the mamba import lets the model load, but the bitsandbytes 4-bit
   quantization collides with the model's custom CUDA kernels during forward pass
3. This causes a 44 GB transient allocation in the MoE layer that OOMs even on
   A100-80GB — the weights aren't truly quantized to 4-bit due to the hook collision

**Solution**: Unsloth has verified day-zero Nemotron-3-Nano-30B support. It handles
the mamba_ssm + bitsandbytes collision internally with its own quantization path.
VRAM requirement: ~24 GB for QLoRA (4-bit, rank 32).

## Constitution Compliance

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Reproducibility | PASS | Config logged, seed=42 |
| II. Compute Efficiency | PASS | A100-80GB, ~24GB VRAM, well within limits |
| III. Iterative | PASS | Single change: training framework (PEFT→Unsloth) |
| IV. Evaluation Rigor | PASS | Will evaluate on held-out val split |
| V. Model Constraint | PASS | Training on exact `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| VI. Open-Source | PASS | Unsloth is Apache 2.0 |
| VII. Ablation | PASS | Treatment variable: training framework |

## Execution Steps

### Step 0: Verify GPU (30 seconds)

```python
import subprocess
r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                     "--format=csv,noheader"], capture_output=True, text=True)
gpu = r.stdout.strip()
assert "A100" in gpu and int(gpu.split(",")[1].strip().split()[0]) >= 70000
print(f"GPU: {gpu}")
```

### Step 1: Install Unsloth + Dependencies (3-5 min)

```bash
pip install -q unsloth
pip install -q --no-deps trl peft accelerate bitsandbytes
```

Unsloth pins its own compatible versions of transformers, peft, etc. Use
`--no-deps` for trl/peft to avoid version conflicts.

### Step 2: Clone Repo + Setup Data (3-5 min)

```python
import os
os.chdir('/content')
os.system("git clone -b 001-llm-reasoning-finetuning "
          "https://github.com/yashashav-dk/homework3_llm_reasoning_finetuning.git repo")
os.chdir('/content/repo')

# Kaggle credentials
from google.colab import userdata
os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')
```

Download + classify + split + solve + trace + format SFT:

```python
# Download competition data
os.system("pip install -q kaggle")
os.system("kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge -p data/")
# ... extract zip, run prepare, splits, solvers, traces, format_sft
# (reuse the inline code from previous successful data pipeline cell)
```

**Checkpoint**: `data/sft/train_sft_full.jsonl` exists with ~6857 examples.

### Step 3: Load Model with Unsloth (2-3 min)

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    max_seq_length=4096,
    dtype=None,         # auto-detect
    load_in_4bit=True,  # QLoRA 4-bit
)
```

**Gate**: Print `torch.cuda.memory_allocated()` — should be ~17-20 GB.

### Step 4: Apply LoRA (< 1 min)

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=16,
    target_modules=["in_proj", "out_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
)
model.print_trainable_parameters()
```

**Gate**: Trainable params should be ~880M / 32.5B (2.71%).

### Step 5: Load SFT Data (< 1 min)

```python
from datasets import Dataset
import json

records = []
with open("data/sft/train_sft_full.jsonl") as f:
    for line in f:
        records.append(json.loads(line))

dataset = Dataset.from_list(records)
print(f"Training examples: {len(dataset)}")
```

**Gate**: Should show 6857 examples.

### Step 6: Train (10-20 min on A100)

```python
from trl import SFTTrainer, SFTConfig

sft_config = SFTConfig(
    output_dir="checkpoints/exp-011-full-sft",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    max_length=4096,               # TRL 1.x parameter name
    num_train_epochs=1,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,    # TRL 1.x parameter name
    train_dataset=dataset,
    args=sft_config,
)

print("Starting training...")
result = trainer.train()
print(f"Training loss: {result.training_loss:.4f}")
```

**Gates**:
- Training starts without OOM
- Loss decreases over steps
- Final loss is finite and < 10.0

### Step 7: Save Adapter (< 1 min)

```python
# Save as standard PEFT adapter (do NOT merge)
model.save_pretrained("checkpoints/exp-011-full-sft")
tokenizer.save_pretrained("checkpoints/exp-011-full-sft")

# Verify files
import os
adapter_dir = "checkpoints/exp-011-full-sft"
files = sorted(os.listdir(adapter_dir))
print(f"Adapter files: {files}")

has_config = "adapter_config.json" in files
has_weights = any("adapter_model" in f for f in files)
assert has_config and has_weights, f"Missing adapter files: {files}"

# Verify adapter_config.json
import json
with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
    cfg = json.load(f)
assert cfg["r"] <= 32, f"LoRA rank {cfg['r']} exceeds competition limit of 32"
print(f"LoRA rank: {cfg['r']}")
print(f"Target modules: {cfg['target_modules']}")

# Fix base_model_name_or_path if needed
if cfg.get("base_model_name_or_path") != "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16":
    cfg["base_model_name_or_path"] = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print("Fixed base_model_name_or_path")

print("\nTRAINING SUCCEEDED")
```

### Step 8: Package Submission (< 1 min)

```python
import zipfile

with zipfile.ZipFile("submissions/exp-011.zip", "w") as z:
    z.write(os.path.join(adapter_dir, "adapter_config.json"), "adapter_config.json")
    # Find the weights file (safetensors or bin)
    for f in files:
        if "adapter_model" in f:
            z.write(os.path.join(adapter_dir, f), f)

# Verify
with zipfile.ZipFile("submissions/exp-011.zip", "r") as z:
    print("Submission contents:")
    for info in z.infolist():
        print(f"  {info.filename}: {info.file_size/1024/1024:.1f} MB")
```

### Step 9: Download Adapter to Local Machine

Download `submissions/exp-011.zip` from Colab to local, then submit to Kaggle:

```bash
kaggle competitions submit \
    -c nvidia-nemotron-model-reasoning-challenge \
    -f submissions/exp-011.zip \
    -m "exp-011: Unsloth QLoRA, all categories, r=32"
```

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Unsloth doesn't support TRL 1.2.0 | Unsloth bundles its own TRL version; use `--no-deps` |
| Unsloth's LoRA adapter format differs | It uses standard PEFT format; verified in docs |
| adapter_config.json has wrong base_model | Step 7 fixes it explicitly |
| Layer merging bug (GitHub #3810) | We do NOT merge — submit adapter directly |
| Colab session timeout | Training is ~15 min; well within limits |
| OOM during loading | Unsloth's custom loader handles this; monitor with nvidia-smi |

## Expected Timeline

| Step | Duration | Cumulative |
|------|----------|------------|
| 0: GPU check | 30s | 30s |
| 1: Install Unsloth | 3 min | 3.5 min |
| 2: Clone + data pipeline | 5 min | 8.5 min |
| 3: Load model | 3 min | 11.5 min |
| 4: Apply LoRA | 30s | 12 min |
| 5: Load data | 30s | 12.5 min |
| 6: Train | 15 min | 27.5 min |
| 7: Save adapter | 30s | 28 min |
| 8: Package | 30s | 28.5 min |
| 9: Download + submit | 2 min | 30.5 min |

**Total: ~30 minutes**

## Files Modified

None — this plan uses Unsloth's API directly in notebook cells, bypassing our
`src/train.py` (which has TRL compat issues). The data pipeline still uses our
`src/` modules. The adapter output is standard PEFT format, same as what
`src/package.py` expects.

## What NOT to Do

- Do NOT install mamba-ssm or causal-conv1d (Unsloth handles this internally)
- Do NOT use `src/train.py` (it has TRL 1.x compat issues; use Unsloth SFTTrainer directly)
- Do NOT merge the adapter into base weights (known bug, competition wants adapter only)
- Do NOT use openNemo-Cascade-2 (different weights, adapter won't transfer correctly)
- Do NOT patch modeling_nemotron_h.py (Unsloth has its own model loading path)
