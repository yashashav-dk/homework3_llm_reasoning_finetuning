# Implementation Plan: LLM Reasoning Fine-Tuning Pipeline

**Branch**: `001-llm-reasoning-finetuning` | **Date**: 2026-04-14 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-llm-reasoning-finetuning/spec.md`
**Competition Deadline**: 2026-06-15

## Summary

Build a QLoRA fine-tuning pipeline for NVIDIA Nemotron-3-Nano-30B
(30B MoE, 3.5B active) to maximize reasoning accuracy on a novel
NVIDIA benchmark. The pipeline spans baseline evaluation, prompt
engineering, CoT data curation from OpenMathReasoning, QLoRA
fine-tuning (rank <= 32), and LoRA adapter packaging for Kaggle
submission. All work runs on a single RTX PRO 6000 (96GB VRAM) on
a Kaggle G4 VM. See [research.md](research.md) for technology
decisions.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: PyTorch >=2.2, Transformers >=4.45,
PEFT >=0.12, TRL >=0.12, vLLM >=0.12, datasets >=3.0,
bitsandbytes >=0.44 (for 4-bit quantization)
**Storage**: File-based (CSV experiment log, YAML configs, JSON
results, model checkpoints, LoRA adapters on disk)
**Testing**: Local held-out validation split (5% OpenMathReasoning,
stratified by difficulty); smoke tests (1-2 batches) before full
runs; local vLLM evaluation matching competition params
**Target Platform**: Kaggle G4 VM (Linux, single NVIDIA RTX PRO
6000 Blackwell, 96GB GDDR7 VRAM)
**Project Type**: ML pipeline (Python scripts + Kaggle notebooks)
**Performance Goals**: Maximize reasoning accuracy (exact match /
numerical tolerance 1e-4) on competition benchmark; answers in
\boxed{} LaTeX format
**Constraints**: Single GPU (96GB VRAM), 12h GPU session limit,
LoRA rank <= 32, max_tokens=7680, max_model_len=8192,
temperature=0.0 (greedy), submission is LoRA adapter zip only
**Scale/Scope**: 306K unique problems (OpenMathReasoning, 3.2M CoT
solutions), 30B parameter MoE model (3.5B active), ~20-24GB VRAM
for QLoRA training

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1
design.*

| Principle | Gate | Status |
|-----------|------|--------|
| I. Reproducibility | All runs log config, seed, results | PASS |
| II. Compute Efficiency | QLoRA ~24GB on 96GB GPU, <12h | PASS |
| III. Iterative Experimentation | Ordered: baseline → prompt → SFT → RL | PASS |
| IV. Evaluation Rigor | 5% held-out split, local vLLM eval | PASS |
| V. Model Constraint | Nemotron-3-Nano-30B, LoRA r<=32 | PASS |
| VI. Open-Source Compliance | OpenMathReasoning CC-BY-4.0 | PASS |
| VII. Ablation Discipline | One variable per experiment | PASS |

**Post-Phase-1 re-check**: All gates PASS. VRAM budget confirmed:
~24GB QLoRA training against 96GB available. LoRA rank 32 is within
competition cap. All training data CC-BY-4.0.

## Project Structure

### Documentation (this feature)

```text
specs/001-llm-reasoning-finetuning/
├── plan.md              # This file
├── spec.md              # Feature specification (clarified)
├── research.md          # Phase 0: technology decisions
├── data-model.md        # Phase 1: entity definitions
├── quickstart.md        # Phase 1: getting started guide
├── contracts/           # Phase 1: interface contracts
│   └── experiment-contract.md
└── tasks.md             # Phase 2: task list (via /speckit.tasks)
```

### Source Code (repository root)

```text
src/
├── evaluate.py              # Local evaluation (vLLM + competition metric)
├── train.py                 # QLoRA fine-tuning with PEFT/TRL
├── package.py               # LoRA adapter → submission.zip packaging
├── data/
│   ├── prepare.py           # OpenMathReasoning curation and filtering
│   └── splits.py            # Validation split creation (5%, stratified)
├── metrics/
│   └── competition.py       # Competition metric (boxed extraction + tolerance)
├── prompts/
│   └── templates.py         # Prompt template management
└── utils/
    ├── config.py            # Config loading and hashing
    ├── logging.py           # Experiment CSV logging
    └── seeds.py             # Deterministic seed management

experiments/
├── experiment_log.csv       # Master experiment tracker (append-only)
├── configs/                 # One YAML per experiment
└── results/                 # One JSON per experiment

prompts/                     # Prompt templates (YAML files)
checkpoints/                 # QLoRA adapter checkpoints
notebooks/
├── error_analysis.ipynb     # Baseline error analysis
└── submission_demo.ipynb    # Adapter packaging demo
submissions/                 # submission.zip files for Kaggle
```

**Structure Decision**: Single-project layout. ML research pipeline
with scripts under `src/`, experiment artifacts under `experiments/`,
and adapter outputs under `checkpoints/`. No web/mobile components.

## Workstream Breakdown

### Workstream 1: Baseline & Analysis (P1)

**Goal**: Establish measurable baseline performance with the
unmodified 30B model.

1. Download `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
2. Create fixed local validation split (5% of OpenMathReasoning,
   stratified by difficulty) — this split is frozen for all future
   experiments
3. Implement competition metric locally (boxed extraction + exact
   match / numerical tolerance 1e-4)
4. Run baseline inference with vLLM using competition parameters
   (temperature=0, top_p=1, max_tokens=7680)
5. Score by domain and overall accuracy
6. Categorize errors: wrong answer, format error (no \boxed{}),
   truncation (hit token limit), empty response
7. Build error analysis notebook

**Output**: `exp-001-baseline` in experiment log, error_analysis.ipynb

### Workstream 2: Prompt Engineering (P2)

**Goal**: Maximize accuracy through prompt optimization alone.
Note: competition uses greedy decoding so prompt directly determines
output — no stochastic variation.

1. Design system prompt enabling chain-of-thought reasoning with
   explicit \boxed{} answer instruction
2. Create domain-specific few-shot examples:
   - Math: algebraic/geometric reasoning ending in \boxed{}
   - Code: step-by-step algorithm tracing with numeric answer
   - Logic: formal deduction chain examples
3. Ablation experiments (one variable each):
   - `exp-002`: CoT system prompt vs. baseline (no prompt)
   - `exp-003`: Add math few-shot examples vs. CoT-only
   - `exp-004`: Add code few-shot examples vs. CoT-only
   - `exp-005`: Combined best prompt elements
4. Benchmark all variants on local validation split using
   competition metric

**Output**: Best prompt template, per-domain accuracy comparison.
Note: the best prompt also becomes the prompt format used during
fine-tuning data preparation.

### Workstream 3: Data Curation & Fine-Tuning (P3)

**Goal**: Improve reasoning through QLoRA supervised fine-tuning.

1. Download and explore OpenMathReasoning CoT solutions
2. Filter dataset:
   - Keep only solutions with correct final answers
   - Filter repetitive reasoning patterns
   - Ensure all solutions end with \boxed{} format
   - Remove solutions exceeding 7680 tokens (competition limit)
   - Prefer shorter correct solutions
   - Balance domain coverage
3. Create 5% stratified held-out validation split BEFORE training
4. Format training data as chat conversations matching best prompt
   template from Workstream 2
5. QLoRA fine-tuning experiments (one variable each):
   - `exp-010`: QLoRA r=16 on filtered CoT data
   - `exp-011`: QLoRA r=32 on filtered CoT data
   - `exp-012`: QLoRA r=32 with learning rate sweep
   - `exp-013`: QLoRA r=32 with dataset size ablation
6. Evaluate each adapter on local split using vLLM + competition
   metric (must match competition inference setup)
7. Select best adapter

**Output**: Best QLoRA adapter checkpoint, training metrics

### Workstream 4: RL Ablation (P4, Optional)

**Goal**: Test whether RL post-training improves the SFT adapter.

1. Take best SFT adapter from Workstream 3
2. Apply GRPO or DPO using correct/incorrect answer pairs as
   reward signal
   - `exp-020`: GRPO on best SFT adapter
   - `exp-021`: DPO on best SFT adapter (if GRPO inconclusive)
3. Evaluate on local split — adopt only if accuracy improves

**Output**: RL-enhanced adapter (if improvement confirmed)

### Workstream 5: Submission Pipeline (P5)

**Goal**: Package best adapter and submit to Kaggle.

1. Package best LoRA adapter as `submission.zip`:
   - adapter_config.json (rank <= 32)
   - adapter_model.safetensors (or .bin)
2. Validate locally: load with vLLM using competition params,
   run on validation split, confirm metric matches expectations
3. Submit to Kaggle
4. Document methods in public notebook (required for prize
   eligibility)

**Output**: submission.zip, public documentation notebook

## VRAM Budget

| Scenario | Est. VRAM | Headroom (96GB) |
|----------|-----------|-----------------|
| bf16 inference (vLLM) | ~60 GB | 36 GB |
| 4-bit inference (vLLM) | ~15-18 GB | 78-81 GB |
| QLoRA training (r=32, bs=4) | ~20-24 GB | 72-76 GB |
| QLoRA training (r=32, bs=8) | ~26-30 GB | 66-70 GB |

All scenarios fit comfortably on the RTX PRO 6000 (96GB).

For local evaluation with vLLM, use bf16 inference (~60GB) to
exactly match competition setup. For training, QLoRA with 4-bit
base is required.

## Competition-Fixed Parameters

These parameters are set by the competition and MUST NOT be changed
in local evaluation to avoid train/eval mismatch:

| Parameter | Value |
|-----------|-------|
| max_lora_rank | 32 |
| max_tokens | 7680 |
| top_p | 1.0 |
| temperature | 0.0 |
| max_num_seqs | 64 |
| gpu_memory_utilization | 0.85 |
| max_model_len | 8192 |

## Complexity Tracking

> No Constitution Check violations to justify. All workstreams
> comply with all 7 principles.
