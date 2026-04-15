# Feature Specification: LLM Reasoning Fine-Tuning Pipeline

**Feature Branch**: `001-llm-reasoning-finetuning`
**Created**: 2026-04-14
**Status**: Draft
**Input**: User description of competition workstreams
**Competition**: NVIDIA Nemotron Model Reasoning Challenge (Kaggle)
**Deadline**: 2026-06-15

## Clarifications

### Session 2026-04-14

- Q: What is the competition's primary evaluation metric? → A: Accuracy on final extracted answer (exact match or numerical tolerance 1e-4), with answers in \boxed{} LaTeX format
- Correction: Base model is Nemotron-3-Nano-30B (30B MoE, 3B active), not the 4B variant
- Correction: Max LoRA rank is 32 (competition-enforced)
- Correction: Submission is a LoRA adapter zip (adapter_config.json required), not a notebook execution
- Correction: Inference is competition-controlled (vLLM, temperature=0.0, top_p=1.0, max_tokens=7680, max_model_len=8192, gpu_memory_utilization=0.85)
- Correction: Inference optimization (majority voting, early stopping, adaptive budgets) is NOT applicable — competition controls inference
- Q: Which training data strategy should be prioritized first? → A: CoT (Chain-of-Thought) solutions from OpenMathReasoning. TIR assumes code execution not available at inference; CoT is safer with greedy decoding and 7680-token budget.
- Q: What precision for LoRA fine-tuning the 30B model? → A: QLoRA (4-bit quantized base + bf16 LoRA adapters). 30B model in bf16 is ~60GB leaving insufficient headroom; 4-bit base reduces to ~17GB, enabling practical training on 96GB VRAM.
- Q: How should the local validation split be constructed? → A: Hold out 5% of OpenMathReasoning problems stratified by difficulty before training. Provides same-distribution ground truth for accurate accuracy measurement with competition metric.
- Q: Should the plan include an RL stage after SFT? → A: SFT first, then RL (GRPO/DPO) as optional ablation experiment. Validate SFT gains before adding RL complexity per constitution principle III (iterative experimentation).

## User Scenarios & Testing

### User Story 1 - Baseline Evaluation (Priority: P1)

Load Nemotron-3-Nano-30B, run inference on competition benchmark,
score results, and categorize errors by domain/type to establish
a baseline.

**Why this priority**: Cannot improve what you cannot measure. Baseline
is the foundation for all subsequent experiments.

**Independent Test**: Run inference on local validation split, produce
a score and error breakdown. Deliverable: error analysis notebook.

**Acceptance Scenarios**:

1. **Given** Nemotron-3-Nano-30B loaded on G4 VM, **When** inference
   runs on full benchmark, **Then** scores are recorded with
   per-domain breakdown
2. **Given** baseline results, **When** error analysis runs, **Then**
   errors are categorized by type (math, code, logic) with examples

---

### User Story 2 - Prompt Engineering (Priority: P2)

Develop and benchmark prompt strategies including system prompts,
few-shot examples, and reasoning modes. Note: the competition uses
temperature=0.0 (greedy decoding) so prompt design directly
determines output.

**Why this priority**: Prompt engineering is zero-cost relative to
fine-tuning and may yield significant gains without training.

**Independent Test**: Compare prompt variants on local validation
split; record accuracy delta vs. baseline.

**Acceptance Scenarios**:

1. **Given** baseline prompt, **When** CoT system prompt applied,
   **Then** accuracy delta is measured and logged
2. **Given** domain-specific few-shot examples, **When** applied per
   domain, **Then** per-domain accuracy is compared to baseline

---

### User Story 3 - Data Curation & Fine-Tuning (Priority: P3)

Curate training data from OpenMathReasoning and other sources, apply
quality filtering, and fine-tune with LoRA (rank <= 32) on G4 VM.
The model MUST output answers in \boxed{} format.

**Why this priority**: Fine-tuning requires validated baseline and
prompt strategy as starting points.

**Independent Test**: Fine-tuned LoRA adapter scores higher than best
prompt-only variant on local validation split.

**Acceptance Scenarios**:

1. **Given** curated dataset, **When** LoRA fine-tuning (rank<=32)
   completes on G4 VM, **Then** training finishes within 12h with
   logged metrics
2. **Given** fine-tuned adapter, **When** evaluated on local split,
   **Then** accuracy exceeds prompt-engineering-only baseline
3. **Given** fine-tuned adapter, **When** packaged as submission.zip,
   **Then** zip contains valid adapter_config.json and weights

---

### User Story 4 - Submission Pipeline (Priority: P4)

Package the best LoRA adapter as a competition-compliant
submission.zip and validate it locally before submitting.

**Why this priority**: Integrates all prior work into a submittable
artifact. Cannot submit without this.

**Independent Test**: submission.zip loads correctly with vLLM using
competition parameters and produces \boxed{} answers.

**Acceptance Scenarios**:

1. **Given** best LoRA adapter, **When** packaged as submission.zip,
   **Then** contains adapter_config.json with rank <= 32
2. **Given** submission.zip, **When** loaded with vLLM using
   competition params, **Then** model generates valid \boxed{} answers

### Edge Cases

- Model exceeds VRAM during inference with long sequences (8192 tokens)
- OpenMathReasoning data has format mismatches with model tokenizer
- LoRA adapter exceeds rank 32 limit
- Model fails to produce \boxed{} answers consistently
- Training on CoT data causes model to exceed max_tokens=7680

## Requirements

### Functional Requirements

- **FR-001**: System MUST load Nemotron-3-Nano-30B and run inference
  on competition benchmark
- **FR-002**: System MUST produce per-domain accuracy scores and error
  categorization
- **FR-003**: System MUST support configurable prompt templates with
  few-shot examples
- **FR-004**: System MUST fine-tune with QLoRA (4-bit base + bf16
  LoRA, rank <= 32) within G4 VM memory constraints
- **FR-005**: System MUST produce answers in \boxed{} LaTeX format
- **FR-006**: System MUST package LoRA adapter as submission.zip with
  adapter_config.json
- **FR-007**: All experiment configs, seeds, and results MUST be
  logged per constitution
- **FR-008**: System MUST evaluate accuracy using competition metric
  (exact match or numerical tolerance 1e-4)

### Key Entities

- **Experiment**: A single run with config, seed, model, and results
- **Prompt Template**: System prompt + few-shot examples + format
  instructions (must elicit \boxed{} answers)
- **Training Dataset**: Curated data with provenance and license info
- **LoRA Adapter**: Fine-tuned adapter weights + adapter_config.json
- **Submission**: Packaged submission.zip for Kaggle

## Success Criteria

### Measurable Outcomes

- **SC-001**: Baseline accuracy established within first 2 experiments
- **SC-002**: Prompt engineering improves accuracy by measurable delta
  over baseline
- **SC-003**: Fine-tuned LoRA adapter exceeds prompt-only best on
  local split
- **SC-004**: submission.zip passes local validation with vLLM
- **SC-005**: All experiments are reproducible from logged configs

## Assumptions

- Kaggle G4 VM with RTX PRO 6000 is available for all runs
- Nemotron-3-Nano-30B (30B MoE, 3B active) fits in GPU memory for
  inference and LoRA training with mixed precision on 96GB VRAM
- OpenMathReasoning dataset is permissively licensed (CC-BY-4.0)
- Competition allows any training framework and approach
- vLLM supports loading LoRA adapters for Nemotron-3-Nano-30B
- RL (GRPO/DPO) is a valid follow-up to SFT but not required for
  initial competitive submission

## Competition Parameters (Fixed)

| Parameter | Value |
|-----------|-------|
| max_lora_rank | 32 |
| max_tokens | 7680 |
| top_p | 1.0 |
| temperature | 0.0 |
| max_num_seqs | 64 |
| gpu_memory_utilization | 0.85 |
| max_model_len | 8192 |
