# Implementation Plan: LLM Reasoning Fine-Tuning Pipeline

**Branch**: `001-llm-reasoning-finetuning` | **Date**: 2026-04-16 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification + deep research from competition intelligence
**Competition Deadline**: 2026-06-15
**Revision**: 3 (rebuilt from deep research findings)

## Summary

Fine-tune Nemotron-3-Nano-30B via QLoRA (rank 32) to solve 7 specific
puzzle categories in the NVIDIA reasoning benchmark. The approach
follows the proven winning strategy: programmatically generate
deterministic chain-of-thought (CoT) traces for each puzzle type,
then SFT the model to reproduce those traces. No RL, no distillation,
no OpenMathReasoning — all training data is generated from the 9500
train.csv puzzles using category-specific Python solvers.

**Target**: >= 0.85 accuracy (competitive with current top 15).
**Floor**: 0.668 from 4 easy categories alone (100% solvable).

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: PyTorch >=2.2, Transformers >=4.45,
PEFT >=0.12, TRL >=0.12, vLLM >=0.12, datasets >=3.0,
bitsandbytes >=0.44
**Storage**: File-based (CSV experiment log, YAML configs, JSON
results, LoRA adapters)
**Testing**: Local evaluation using competition metric (extract_
final_answer + verify); held-out split from train.csv
**Target Platform**: Kaggle G4 VM (Linux, RTX PRO 6000, 96GB VRAM)
**Project Type**: ML pipeline (Python solvers + trace generators +
QLoRA training + adapter packaging)
**Performance Goals**: >= 0.85 accuracy on competition benchmark
**Constraints**: Single GPU (96GB VRAM), 12h session, LoRA rank <= 32,
max_tokens=7680, temperature=0.0, submission is LoRA adapter zip
**Scale/Scope**: 9500 train puzzles across 7 categories, 30B MoE
model (3.5B active), ~24GB VRAM for QLoRA training

## Constitution Check

| Principle | Gate | Status |
|-----------|------|--------|
| I. Reproducibility | All runs log config, seed, results | PASS |
| II. Compute Efficiency | QLoRA ~24GB on 96GB GPU, <12h | PASS |
| III. Iterative Experimentation | Category-by-category, easy→hard | PASS |
| IV. Evaluation Rigor | Local eval with exact competition metric | PASS |
| V. Model Constraint | Nemotron-3-Nano-30B, LoRA r<=32 | PASS |
| VI. Open-Source Compliance | train.csv CC-BY-4.0, no distillation | PASS |
| VII. Ablation Discipline | One category/variable per experiment | PASS |

## Project Structure

### Documentation

```text
specs/001-llm-reasoning-finetuning/
├── plan.md                  # This file
├── spec.md                  # Feature specification (clarified)
├── research.md              # Technology decisions
├── deep-research.md         # Competition intelligence
├── competition-reference.md # Metric source, submission demo
├── data-model.md            # Entity definitions
├── quickstart.md            # Getting started guide
├── contracts/               # Interface contracts
└── tasks.md                 # Task list (via /speckit.tasks)
```

### Source Code

```text
src/
├── solvers/                     # Category-specific puzzle solvers
│   ├── base.py                  # Abstract solver interface
│   ├── numeral.py               # Roman numeral conversion
│   ├── gravity.py               # d=0.5gt^2 derivation
│   ├── unit_conversion.py       # Linear scaling
│   ├── cipher.py                # Substitution cipher cracking
│   ├── bit_manipulation.py      # Per-bit boolean function search
│   ├── equation.py              # Operator discovery (4x32 combos)
│   └── cryptarithm.py           # Verbal arithmetic (hardest)
├── trace_generators/            # CoT trace generation per category
│   ├── base.py                  # Abstract trace generator
│   ├── numeral_traces.py        # Roman numeral step-by-step traces
│   ├── gravity_traces.py        # Rate derivation traces
│   ├── unit_conversion_traces.py
│   ├── cipher_traces.py         # Char-by-char decryption traces
│   ├── bit_manipulation_traces.py # Bit-serial gate computation
│   ├── equation_traces.py       # Operator scan traces
│   └── cryptarithm_traces.py
├── data/
│   ├── prepare.py               # Load train.csv, classify categories
│   ├── splits.py                # Train/validation split
│   └── format_sft.py            # Format traces as chat conversations
├── metrics/
│   └── competition.py           # Exact competition metric replica
├── train.py                     # QLoRA fine-tuning with PEFT/TRL
├── evaluate.py                  # vLLM-based evaluation
├── package.py                   # LoRA adapter -> submission.zip
└── utils/
    ├── config.py                # Config loading and hashing
    ├── logging.py               # Experiment CSV logging
    └── seeds.py                 # Deterministic seed management

experiments/
├── experiment_log.csv           # Master tracker
├── configs/                     # YAML per experiment
└── results/                     # JSON per experiment

checkpoints/                     # QLoRA adapter checkpoints
submissions/                     # submission.zip files
notebooks/
├── error_analysis.ipynb         # Category-level error analysis
└── submission_demo.ipynb        # Adapter packaging demo
```

**Structure Decision**: Solver-first architecture. Each puzzle
category gets a solver (produces the answer) and a trace generator
(produces the step-by-step CoT that teaches the model HOW to solve
it). Training data is generated programmatically, not from external
datasets.

## Workstream Breakdown

### Workstream 1: Infrastructure & Baseline (P1)

**Goal**: Set up project, implement competition metric, establish
baseline accuracy of unmodified model.

1. Create project directory structure
2. Implement exact competition metric (`extract_final_answer` +
   `verify` from metric source code)
3. Load train.csv, classify each row into its puzzle category
4. Create train/validation split (90/10 stratified by category)
5. Run baseline inference with vLLM (`enable_thinking=True`,
   competition params)
6. Score baseline by category and overall
7. Error analysis notebook

**Output**: Baseline accuracy per category, error analysis

### Workstream 2: Easy Category Solvers & Traces (P2)

**Goal**: Achieve 100% solve rate on 4 easy categories = 66.8% floor.

**2a. Roman Numeral** (1576 samples, target: 100%)
1. Write numeral solver (enumerate 1-100 Roman numerals)
2. Write trace generator: step-by-step decomposition -> conversion
3. Verify solver produces 100% correct answers on train.csv

**2b. Gravity** (1597 samples, target: 100%)
1. Write gravity solver (derive rate = d/t^2 from examples, apply)
2. Write trace generator: rate-first decomposition, multi-step math,
   rate consistency verification, format to X.XX
3. Verify solver produces 100% correct answers

**2c. Unit Conversion** (1594 samples, target: 100%)
1. Write unit_conversion solver (derive factor = out/in, apply)
2. Write trace generator: same structure as gravity but linear
3. Verify solver produces 100% correct answers

**2d. Cipher** (1576 samples, target: 100%)
1. Write cipher solver (extract char mappings, vocab fill for gaps)
2. Write trace generator: char-by-char decryption, vocabulary
   matching from ~90 Wonderland words
3. Verify solver produces 100% correct answers

**Output**: 4 solvers + 4 trace generators, all verified 100% on
train.csv. Total: 6343/9500 = 66.8% floor.

### Workstream 3: First SFT Run (P3)

**Goal**: Train on easy category traces and verify the model learns.

1. Generate all CoT traces for easy categories (format with
   `<think>...</think>` and `\boxed{}`)
2. Format as chat conversations using model's chat template
3. Verify all traces fit within 7680 token limit
4. QLoRA fine-tuning (r=32, target modules:
   `in_proj|out_proj|up_proj|down_proj`, lora_alpha=16)
5. Evaluate on validation split with competition metric
6. Verify near-100% on easy categories
7. Package and submit to Kaggle for baseline score

**Output**: First LoRA adapter, first Kaggle submission, ~0.67 score

### Workstream 4: Hard Category Solvers (P4)

**Goal**: Add bit manipulation and equation solving for ~0.85 total.

**4a. Bit Manipulation** (1602 samples, target: 85%)
1. Write bit_manipulation solver: per-bit boolean function search
   through 52 gate types (constants -> identity -> NOT -> 2-input ->
   3-input -> 4-input gates)
2. Write trace generator: bit-serial gate computation (spell out
   each operation one bit at a time: `0&1=0 1&1=1`)
3. Include verification step (check candidate against test input)
4. Verify solver on train.csv (expect ~85% solve rate)

**4b. Equation** (732 samples, target: 76-90%)
1. Write equation solver: 4 operand transforms x 32 operators,
   frequency-ordered brute force scan
2. Write trace generator: parse -> scan -> lock -> apply -> answer
3. Include EX2 verification to catch coincidental matches
4. Verify solver on train.csv

**4c. Cryptarithm** (823 samples, target: ~8%)
1. Write cryptarithm solver: detect concatenation/reverse concat
2. Write trace generator for solvable subset
3. Accept low solve rate — this is the hardest category

**Output**: 3 more solvers + trace generators

### Workstream 5: Full SFT & Optimization (P5)

**Goal**: Train on all 7 categories, optimize, submit best adapter.

1. Generate traces for all categories (easy + hard)
2. QLoRA fine-tuning on full trace set
3. Evaluate per-category and overall accuracy
4. Ablation experiments (one variable per experiment):
   - `exp-010`: Easy categories only (baseline)
   - `exp-011`: + bit manipulation traces
   - `exp-012`: + equation traces
   - `exp-013`: + cryptarithm traces
   - `exp-014`: Trace format variations (if accuracy <0.85)
   - `exp-015`: Learning rate / epoch ablation
5. Inspect minimum logprob per trace — identify weak spots
6. Refine traces where model struggles (tokenization issues,
   arithmetic steps too complex)
7. Select best adapter, package submission.zip
8. Submit to Kaggle

**Output**: Best LoRA adapter, competitive submission

### Workstream 6: Polish & Documentation (P6)

**Goal**: Optimize for final leaderboard, publish methodology.

1. Iterate on trace quality for categories with <100% solve rate
2. Try RL (GRPO) as optional ablation on best SFT adapter
3. Create public documentation notebook (required for prizes)
4. Select 2 final submissions
5. Final submission before June 15 deadline

## VRAM Budget

| Scenario | Est. VRAM | Headroom (96GB) |
|----------|-----------|-----------------|
| bf16 inference (vLLM) | ~60 GB | 36 GB |
| QLoRA training (r=32, bs=4) | ~20-24 GB | 72-76 GB |
| Trace generation (CPU) | 0 GB | 96 GB |
| Solver verification (CPU) | 0 GB | 96 GB |

Most work (solvers, trace generation) is CPU-only Python. GPU is
needed only for inference evaluation and QLoRA training.

## Competition-Fixed Parameters

| Parameter | Value |
|-----------|-------|
| max_lora_rank | 32 |
| max_tokens | 7680 |
| top_p | 1.0 |
| temperature | 0.0 |
| max_num_seqs | 64 |
| gpu_memory_utilization | 0.85 |
| max_model_len | 8192 |
| enable_thinking | True |

## LoRA Configuration (from official submission demo)

```python
LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

## Accuracy Projections

| Stage | Categories Solved | Projected Accuracy |
|-------|-------------------|-------------------|
| Baseline (no training) | - | ~0.49 |
| WS3: Easy categories | numeral+gravity+unit+cipher | ~0.67 |
| WS5: + bit manipulation | + 85% of 1602 | ~0.81 |
| WS5: + equation | + 80% of 732 | ~0.87 |
| WS5: + cryptarithm | + 8% of 823 | ~0.88 |

## Complexity Tracking

> No Constitution Check violations. Strategy is simpler than original
> plan — no RL, no external datasets, no inference optimization.
> All complexity is in understanding the puzzle types.
