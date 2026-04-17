# Quickstart: LLM Reasoning Fine-Tuning Pipeline

**Branch**: `001-llm-reasoning-finetuning` | **Date**: 2026-04-16
**Revision**: 3 (solver-first architecture)

## Prerequisites

- Kaggle account with G4 VM access
- Python 3.10+
- Competition data (train.csv) downloaded

## 1. Environment Setup

```bash
git clone <repo-url> && cd homework3_llm_reasoning_finetuning
pip install -r requirements.txt
```

## 2. Classify Puzzles

```bash
python src/data/prepare.py \
    --input data/train.csv \
    --output data/puzzles_classified.jsonl
```

This classifies each puzzle into one of 7 categories:
numeral, gravity, unit_conversion, cipher, bit_manipulation,
equation_numeric_deduce/guess, cryptarithm_deduce/guess.

## 3. Create Train/Validation Split

```bash
python src/data/splits.py \
    --input data/puzzles_classified.jsonl \
    --output data/splits/ \
    --val-pct 0.10 \
    --stratify category \
    --seed 42
```

## 4. Run Solvers (verify correctness)

```bash
# Run all solvers on train.csv
python -m src.solvers.numeral --input data/puzzles_classified.jsonl
python -m src.solvers.gravity --input data/puzzles_classified.jsonl
python -m src.solvers.unit_conversion --input data/puzzles_classified.jsonl
python -m src.solvers.cipher --input data/puzzles_classified.jsonl
python -m src.solvers.bit_manipulation --input data/puzzles_classified.jsonl
python -m src.solvers.equation --input data/puzzles_classified.jsonl
python -m src.solvers.cryptarithm --input data/puzzles_classified.jsonl
```

Each solver prints its accuracy on train.csv.

## 5. Generate CoT Traces

```bash
# Generate traces for all solvable puzzles
python -m src.trace_generators.numeral_traces --output data/traces/
python -m src.trace_generators.gravity_traces --output data/traces/
# ... (one per category)
```

## 6. Format SFT Data

```bash
python src/data/format_sft.py \
    --traces data/traces/ \
    --split data/splits/train_ids.json \
    --output data/sft/train_sft.jsonl \
    --max-tokens 7680
```

## 7. QLoRA Fine-Tuning

```bash
python src/train.py \
    --config experiments/configs/exp-010-easy.yaml \
    --seed 42
```

## 8. Evaluate

```bash
python src/evaluate.py \
    --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --adapter checkpoints/exp-010-easy/ \
    --split data/splits/val_ids.json \
    --seed 42
```

## 9. Package & Submit

```bash
python src/package.py \
    --adapter checkpoints/best/ \
    --output submissions/submission.zip
```

Upload `submission.zip` to Kaggle.

## Workflow: Adding a New Category

1. Write solver in `src/solvers/{category}.py`
2. Verify solver accuracy on train.csv
3. Write trace generator in `src/trace_generators/{category}_traces.py`
4. Generate traces, format SFT data, retrain, evaluate
5. One category per experiment (ablation discipline)
