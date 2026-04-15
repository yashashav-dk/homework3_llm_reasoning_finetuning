<!--
  Sync Impact Report
  ==================
  Version change: 0.0.0 (template) → 1.0.0
  Modified principles: N/A (initial population)
  Added sections:
    - 7 Core Principles (I–VII)
    - Hardware & Environment Constraints
    - Experiment Workflow
    - Governance
  Removed sections: None
  Templates requiring updates:
    - .specify/templates/plan-template.md — ✅ compatible (Constitution Check
      section already generic)
    - .specify/templates/spec-template.md — ✅ compatible (no principle-specific
      references)
    - .specify/templates/tasks-template.md — ✅ compatible (phase structure is
      generic)
  Follow-up TODOs: None
-->
# LLM Reasoning Fine-Tuning Constitution

## Core Principles

### I. Reproducibility

Every experiment MUST be fully reproducible from its recorded
configuration alone.

- All training runs MUST log: model checkpoint, hyperparameters,
  random seeds, dataset version, and hardware environment.
- Configuration files (YAML/JSON) MUST be committed alongside
  results before any conclusions are drawn.
- Random seeds MUST be set explicitly for PyTorch, NumPy, and
  Python's `random` module at the start of every script.
- Results MUST be recorded in a structured experiment log
  (e.g., CSV, JSON, or Weights & Biases) with run ID, config
  hash, and metric values.

**Rationale**: Without reproducibility, no result can be trusted
or built upon. A gain that cannot be replicated is not a gain.

### II. Compute Efficiency

All solutions MUST fit within the Kaggle G4 VM hardware envelope.

- Target hardware: single NVIDIA RTX PRO 6000 GPU, Kaggle G4 VM.
- Training scripts MUST complete within Kaggle's per-session time
  limits (typically 12 hours GPU, 9 hours for notebooks).
- Peak GPU memory usage MUST stay below the RTX PRO 6000's VRAM
  capacity with a safety margin (use gradient checkpointing,
  mixed precision, or reduced batch sizes as needed).
- Before committing to a full training run, a short smoke test
  (1–2 batches) MUST confirm the pipeline fits in memory and
  runs without errors.

**Rationale**: Solutions that cannot run on competition hardware
are invalid. Efficiency is a hard constraint, not a nice-to-have.

### III. Iterative Experimentation

Prefer small, validated improvements over large speculative
changes.

- Each experiment MUST target a single hypothesis with a clear
  expected outcome documented before the run begins.
- Changes MUST be incremental: one modification at a time from a
  known-good baseline.
- After each experiment, results MUST be compared to the baseline
  and the decision to adopt or revert MUST be recorded.
- Speculative multi-variable changes are NOT permitted unless
  each variable has been independently validated first.

**Rationale**: Large jumps obscure what actually works. Small
steps with clear signals compound into reliable progress.

### IV. Evaluation Rigor

Every model MUST be validated against a held-out local split
before any competition submission.

- A fixed local validation split MUST be created at project
  start and MUST NOT change across experiments.
- The local split MUST be representative of the competition's
  evaluation distribution (stratified by difficulty, category,
  or other relevant dimensions).
- Models MUST meet or exceed the current local baseline score
  before a submission is made.
- Overfitting to the public leaderboard is prohibited; local
  validation is the primary decision metric.

**Rationale**: Submitting without local validation wastes limited
submission attempts and encourages leaderboard chasing over
genuine model improvement.

### V. Model Constraint

All fine-tuning work MUST use NVIDIA Nemotron-3-Nano-30B as
the base model.

- The base model MUST be `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
  (30B total parameters, 3.5B active, Mamba-2 + Transformer + MoE).
- No other base models (e.g., LLaMA, Mistral, Qwen) are
  permitted, even for comparison baselines.
- LoRA adapters MUST have rank <= 32 (competition-enforced).
- QLoRA (4-bit quantized base) is permitted and recommended
  for training within GPU memory constraints.
- Model merging with non-Nemotron checkpoints is NOT permitted.
- The final submission MUST be a LoRA adapter (submission.zip
  with adapter_config.json), not a full model.

**Rationale**: Competition rules mandate this model. Any solution
using a different base is ineligible regardless of performance.

### VI. Open-Source Compliance

All datasets, code, and dependencies MUST use permissive
open-source licenses.

- Training data MUST be sourced from datasets with licenses
  that permit commercial and competition use (Apache 2.0, MIT,
  CC-BY, CC-BY-SA, or public domain).
- Third-party code and libraries MUST carry permissive licenses
  compatible with competition rules.
- Any dataset used MUST have its license documented in the
  experiment log before training begins.
- Proprietary or restrictively licensed resources (including
  model-generated synthetic data from closed APIs, unless
  explicitly allowed) MUST NOT be used.

**Rationale**: License violations can disqualify an entry and
create legal liability. Compliance MUST be verified upfront,
not retroactively.

### VII. Ablation Discipline

Exactly one variable MUST change per experiment to isolate
causal effects.

- Each experiment MUST have a documented control (the previous
  best configuration) and a single treatment variable.
- If multiple changes are necessary (e.g., a new dataset
  requires a new tokenizer), this MUST be documented as a
  compound change with explicit justification.
- Ablation results MUST be recorded in a comparison table
  showing: variable changed, control score, treatment score,
  delta, and statistical significance (where applicable).
- Negative results MUST be logged with the same rigor as
  positive results.

**Rationale**: Without isolation, improvements cannot be
attributed and regressions cannot be diagnosed. Ablation
discipline is the foundation of systematic progress.

## Hardware & Environment Constraints

- **GPU**: Single NVIDIA RTX PRO 6000 (Kaggle G4 VM)
- **Runtime**: Kaggle notebook sessions (12h GPU limit)
- **Framework**: PyTorch with Hugging Face Transformers/TRL
- **Precision**: Mixed precision (fp16/bf16) MUST be used to
  maximize effective VRAM
- **Dependencies**: All pip/conda packages MUST be pinned to
  exact versions in `requirements.txt`
- **Offline Mode**: Final submission MUST run with no internet
  access (Kaggle submission environment constraint)

## Experiment Workflow

1. **Hypothesize**: Document the change and expected outcome
   before running anything.
2. **Configure**: Create or modify a config file; commit it.
3. **Smoke Test**: Run 1–2 batches to verify memory and
   correctness.
4. **Train**: Execute the full run with logging enabled.
5. **Evaluate**: Compare against local validation split and
   record metrics.
6. **Decide**: Adopt (new baseline) or revert (log negative
   result). Never leave an experiment in an ambiguous state.
7. **Submit**: Only after local validation confirms improvement
   over the current best.

## Governance

- This constitution is the authoritative guide for all
  development decisions in this project. It supersedes ad-hoc
  practices and informal agreements.
- Amendments require: (1) written justification documenting why
  the change is needed, (2) review of impact on existing
  experiment logs, and (3) version bump following semver rules.
- Versioning follows Semantic Versioning:
  - MAJOR: Principle removed or fundamentally redefined.
  - MINOR: New principle or section added, or material expansion.
  - PATCH: Clarifications, wording, or formatting fixes.
- All experiment designs MUST be checked against this
  constitution before execution begins.
- Compliance violations discovered post-hoc MUST be flagged and
  the affected results quarantined until re-validated.

**Version**: 1.1.0 | **Ratified**: 2026-04-14 | **Last Amended**: 2026-04-14
