# Plan: Colab Pro Automation via Playwright

**Date**: 2026-04-21
**Goal**: Execute GPU-blocked tasks (T052-T054) on Google Colab Pro L4 using Playwright MCP
**Prerequisite**: User is logged into Google Colab in the browser

## Context

Tasks T052-T054 require a GPU (Colab Pro L4, 24 GB VRAM) for:
- T052: QLoRA training (~25-35 min)
- T053: vLLM evaluation (~15-25 min)
- T054: End-to-end notebook validation

The notebook `notebooks/colab_pipeline.ipynb` is already built as a thin
orchestrator. But cells 6-7 (solvers + traces) and cell 8 (format_sft) have
already been run locally — so we only need to:
1. Upload the pre-built SFT data (`data/sft/train_sft_full.jsonl`)
2. Upload the repo code (or clone from git)
3. Run training (cell 9)
4. Run evaluation (cells 10-11)
5. Download the adapter + results

## Approach

Use Playwright MCP to drive Google Colab's web UI directly. Colab cells are
executed via the browser — no API needed.

## Steps

### Step 0: Verify Colab Session

1. Navigate to `https://colab.research.google.com/`
2. Take snapshot to verify user is logged in (look for profile icon / "New notebook" button)
3. If not logged in → STOP, ask user to log in

### Step 1: Create New Notebook & Set GPU Runtime

1. Click "New notebook" or navigate to new notebook URL
2. Go to Runtime → Change runtime type
3. Select "L4" GPU (or "T4" as fallback)
4. Click Save
5. Verify GPU by running: `!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader`
6. Assert output contains "L4" (24 GB) or "T4" (16 GB)
7. If no GPU or wrong GPU → STOP, report issue

### Step 2: Install Dependencies & Clone Repo

Run in a single cell:
```python
%%bash
pip install -q torch>=2.2.0 transformers>=4.45.0 peft>=0.12.0 trl>=0.12.0 \
    vllm>=0.12.0 datasets>=3.0.0 accelerate>=1.0.0 bitsandbytes>=0.44.0 \
    pyyaml>=6.0
```

Then clone repo:
```python
import os, sys
!git clone https://github.com/{ACTUAL_REPO_URL}.git /content/repo
os.chdir('/content/repo')
sys.path.insert(0, '/content/repo')
```

### Step 3: Upload Pre-built SFT Data

Instead of re-running solvers + traces + format_sft on Colab (wastes GPU time),
upload the locally-built `data/sft/train_sft_full.jsonl` (6857 examples, ~4 MB).

Options:
- a) Use Playwright file upload to Colab's file browser
- b) Upload to Google Drive, mount in Colab
- c) Push data to git repo, pull in Colab

**Chosen**: Option (a) — direct file upload via Colab sidebar. Simplest, no
external dependencies.

Steps:
1. Click the "Files" icon in Colab sidebar (folder icon)
2. Create directory structure: `!mkdir -p data/sft experiments/configs checkpoints submissions`
3. Upload `data/sft/train_sft_full.jsonl` via file upload
4. Upload `experiments/configs/exp-011-full-sft.yaml` (or verify it's in the cloned repo)
5. Verify: `!wc -l data/sft/train_sft_full.jsonl` should show 6857

### Step 4: Run QLoRA Training (T052)

Execute in a cell:
```python
!python -m src.train --config experiments/configs/exp-011-full-sft.yaml
```

**Monitoring**:
- Wait for "Starting training" log line
- Watch for collator mask % (expect 40-70%)
- Watch for training loss (should decrease, stay finite)
- Wait for "Saving LoRA adapter" line
- Expected duration: ~25-35 min
- Poll every 60s by checking cell output for completion indicators

**Success criteria**:
- "Adapter validation passed" in output
- `checkpoints/exp-011-full-sft/adapter_config.json` exists
- `checkpoints/exp-011-full-sft/adapter_model.safetensors` exists

**Failure handling**:
- OOM → retry with `per_device_train_batch_size: 2` (modify config inline)
- Loss NaN → report and stop

### Step 5: Run vLLM Evaluation (T053)

First, free VRAM:
```python
import torch, gc
gc.collect()
torch.cuda.empty_cache()
```

Then evaluate:
```python
!python -m src.evaluate \
    --adapter checkpoints/exp-011-full-sft \
    --config experiments/configs/exp-011-full-sft.yaml \
    --output experiments/results/exp-011.json
```

**Monitoring**:
- Wait for vLLM model loading
- Wait for inference completion
- Wait for accuracy table
- Expected duration: ~15-25 min

**Success criteria**:
- Per-category accuracy table printed
- Overall accuracy > 0.49 (beats baseline)
- `experiments/results/exp-011.json` written

### Step 6: Package & Download Results (T055)

```python
!python -m src.package \
    --adapter checkpoints/exp-011-full-sft \
    --output submissions/exp-011.zip
```

Then download:
- `submissions/exp-011.zip` (adapter for Kaggle submission)
- `experiments/results/exp-011.json` (accuracy results)
- Training logs from output

### Step 7: Submit to Kaggle (T056, optional)

If Kaggle credentials are available in Colab:
```python
!pip install -q kaggle
!kaggle competitions submit \
    -c nvidia-nemotron-model-reasoning-challenge \
    -f submissions/exp-011.zip \
    -m "exp-011: full SFT, all categories, gradient checkpointing"
```

## Playwright Execution Strategy

### Cell Execution Pattern

Colab cells are executed by:
1. Clicking into the cell (or creating a new code cell)
2. Typing/pasting code
3. Pressing Shift+Enter or clicking the Run button
4. Waiting for the cell to complete (spinner disappears, output appears)

For long-running cells (training ~30 min, eval ~20 min):
- Use `browser_wait_for` with generous timeouts
- Poll cell output periodically via `browser_snapshot`
- Look for completion markers in output text

### Error Recovery

| Error | Detection | Recovery |
|-------|-----------|----------|
| OOM during training | "OutOfMemoryError" in cell output | Modify batch_size to 2, re-run |
| vLLM OOM at eval | "OutOfMemoryError" in cell output | Restart runtime, re-run eval only (adapter is saved) |
| Session disconnect | Page shows "Reconnect" dialog | Click reconnect, re-run from last incomplete step |
| Wrong GPU | nvidia-smi shows no L4 | Change runtime type, restart |
| Import error | ModuleNotFoundError in output | Re-run pip install cell |

## Timeline

| Step | Duration | Cumulative |
|------|----------|------------|
| 0-2: Setup + deps | ~5 min | 5 min |
| 3: Upload data | ~2 min | 7 min |
| 4: Training | ~25-35 min | 32-42 min |
| 5: Evaluation | ~15-25 min | 47-67 min |
| 6-7: Package + submit | ~3 min | 50-70 min |

**Total**: ~50-70 min (well within 2h budget)

## Critical Files

- `data/sft/train_sft_full.jsonl` (upload from local, 6857 examples)
- `experiments/configs/exp-011-full-sft.yaml` (in repo)
- `src/train.py` (in repo)
- `src/evaluate.py` (in repo)
- `src/package.py` (in repo)
