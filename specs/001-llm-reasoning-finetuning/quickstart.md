# Quickstart: LLM Reasoning Fine-Tuning Pipeline

**Branch**: `001-llm-reasoning-finetuning` | **Date**: 2026-04-14
**Revision**: 2 (corrected for 30B model, adapter submission)

## Prerequisites

- Kaggle account with G4 VM (RTX PRO 6000, 96GB VRAM) access
- Python 3.10+
- HuggingFace account (for model download)

## 1. Environment Setup

```bash
# Clone the repository
git clone <repo-url> && cd homework3_llm_reasoning_finetuning

# Install dependencies
pip install -r requirements.txt
```

Required packages (pin versions in requirements.txt):
```text
torch>=2.2.0
transformers>=4.45.0
peft>=0.12.0
trl>=0.12.0
vllm>=0.12.0
datasets>=3.0.0
accelerate>=1.0.0
bitsandbytes>=0.44.0
wandb  # optional, for development tracking
```

## 2. Download Model and Data

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Download Nemotron-3-Nano-30B (bf16 for inference)
model_id = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# For QLoRA training, load in 4-bit
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

# Download OpenMathReasoning CoT solutions
dataset = load_dataset(
    "nvidia/OpenMathReasoning",
    split="train",
    streaming=True  # 49.5GB — stream to avoid disk issues
)
```

## 3. Create Validation Split

```bash
# Create 5% stratified held-out split (run ONCE, freeze forever)
python src/data/splits.py \
    --source nvidia/OpenMathReasoning \
    --solution-type cot \
    --holdout-pct 0.05 \
    --stratify-by difficulty \
    --seed 42 \
    --output data/splits/
```

## 4. Run Baseline Inference

```bash
# Start vLLM server with competition parameters
python -m vllm.entrypoints.openai.api_server \
    --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 64

# In another terminal: run baseline evaluation
python src/evaluate.py \
    --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --split data/splits/validation.jsonl \
    --temperature 0.0 \
    --top-p 1.0 \
    --max-tokens 7680 \
    --output experiments/results/exp-001-baseline.json \
    --seed 42
```

## 5. Run a Prompt Engineering Experiment

```bash
# Test CoT prompt variant
python src/evaluate.py \
    --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --prompt-template prompts/cot-boxed.yaml \
    --split data/splits/validation.jsonl \
    --temperature 0.0 \
    --top-p 1.0 \
    --max-tokens 7680 \
    --output experiments/results/exp-002-cot-prompt.json \
    --seed 42
```

## 6. Fine-Tune with QLoRA

```bash
# QLoRA fine-tuning on curated CoT data
python src/train.py \
    --config experiments/configs/exp-010-qlora-r32.yaml \
    --seed 42
```

Sample config (`experiments/configs/exp-010-qlora-r32.yaml`):
```yaml
model_name: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
quantization: nf4
lora_rank: 32
lora_alpha: 64
lora_target_modules: [q_proj, k_proj, v_proj, o_proj]
learning_rate: 2e-4
batch_size: 4
gradient_accumulation: 8
max_seq_length: 2048
num_epochs: 1
warmup_ratio: 0.1
seed: 42
dataset_id: openmathreas-cot-filtered-v1
prompt_template: prompts/cot-boxed.yaml
treatment_variable: lora_rank=32
control_run_id: exp-002-cot-prompt
```

## 7. Evaluate Fine-Tuned Adapter

```bash
# Evaluate with vLLM + LoRA adapter (matches competition setup)
python src/evaluate.py \
    --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --adapter checkpoints/exp-010-qlora-r32/ \
    --split data/splits/validation.jsonl \
    --temperature 0.0 \
    --top-p 1.0 \
    --max-tokens 7680 \
    --output experiments/results/exp-010-qlora-r32.json \
    --seed 42
```

## 8. Package and Submit

```bash
# Package best adapter as submission.zip
python src/package.py \
    --adapter checkpoints/best/ \
    --output submissions/submission.zip

# Verify the zip contains required files
unzip -l submissions/submission.zip
# Must show: adapter_config.json, adapter_model.safetensors (or .bin)
```

Upload `submission.zip` to Kaggle competition page.

## Experiment Workflow (Constitution-Compliant)

1. Copy an existing config and change exactly **one** variable
2. Document your hypothesis in the experiment log
3. Run smoke test (2 batches) to verify memory fit
4. Run full experiment with `--seed 42`
5. Evaluate with vLLM using **competition parameters exactly**
6. Compare results to control run using competition metric
7. Record decision: adopt or revert
8. Commit config + results before moving to next experiment

## Directory Structure

```text
homework3_llm_reasoning_finetuning/
├── src/
│   ├── evaluate.py          # vLLM-based evaluation pipeline
│   ├── train.py             # QLoRA fine-tuning (PEFT/TRL)
│   ├── package.py           # Adapter → submission.zip
│   ├── data/
│   │   ├── prepare.py       # OpenMathReasoning curation
│   │   └── splits.py        # Validation split creation
│   ├── metrics/
│   │   └── competition.py   # \boxed{} extraction + accuracy
│   ├── prompts/
│   │   └── templates.py     # Prompt template management
│   └── utils/
│       ├── config.py        # Config loading/hashing
│       ├── logging.py       # Experiment CSV logging
│       └── seeds.py         # Seed management
├── experiments/
│   ├── experiment_log.csv   # Master experiment tracker
│   ├── configs/             # YAML configs per experiment
│   └── results/             # JSON results per experiment
├── prompts/                 # Prompt templates (YAML)
├── checkpoints/             # QLoRA adapter checkpoints
├── data/
│   └── splits/              # Frozen validation split
├── notebooks/
│   ├── error_analysis.ipynb # Baseline error analysis
│   └── submission_demo.ipynb# Adapter packaging demo
├── submissions/             # submission.zip files
├── requirements.txt         # Pinned dependencies
└── specs/                   # Specify planning artifacts
```
