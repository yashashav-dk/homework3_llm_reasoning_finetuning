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
- Correction: Base model is Nemotron-3-Nano-30B (30B MoE, 3.5B active), not the 4B variant
- Correction: Max LoRA rank is 32 (competition-enforced)
- Correction: Submission is a LoRA adapter zip (adapter_config.json required), not a notebook execution
- Correction: Inference is competition-controlled (vLLM, temperature=0.0, top_p=1.0, max_tokens=7680, max_model_len=8192, gpu_memory_utilization=0.85)
- Correction: Inference optimization (majority voting, early stopping, adaptive budgets) is NOT applicable — competition controls inference
- Q: Which training data strategy should be prioritized first? → A: CoT (Chain-of-Thought) solutions from OpenMathReasoning. TIR assumes code execution not available at inference; CoT is safer with greedy decoding and 7680-token budget.
- Q: What precision for LoRA fine-tuning the 30B model? → A: QLoRA (4-bit quantized base + bf16 LoRA adapters). 30B model in bf16 is ~60GB leaving insufficient headroom; 4-bit base reduces to ~17GB, enabling practical training on 96GB VRAM.
- Q: How should the local validation split be constructed? → A: Hold out 5% of OpenMathReasoning problems stratified by difficulty before training. Provides same-distribution ground truth for accurate accuracy measurement with competition metric.
- Q: Should the plan include an RL stage after SFT? → A: SFT first, then RL (GRPO/DPO) as optional ablation experiment. Validate SFT gains before adding RL complexity per constitution principle III (iterative experimentation).

### Session 2026-04-16 (from competition metric source & model card)

- Correction: Numerical tolerance in metric verify() is 1e-2 (not
  1e-4 as overview text states). Code is authoritative.
- Correction: Binary strings (`[01]+`) are compared strictly with no
  leading-zero tolerance in metric verify(). Model must produce exact
  binary output for bit manipulation puzzles.
- Correction: Competition uses `enable_thinking=True` — model produces
  `<think>...</think>` reasoning traces during evaluation. Training
  must account for this.
- Correction: LoRA target modules must use regex pattern
  `r".*\.(in_proj|out_proj|up_proj|down_proj)$"` per the official
  submission demo (not q_proj/k_proj/v_proj/o_proj).
- Correction: Benchmark domain is novel logical puzzles (bit
  manipulation rules, encryption rules, algebraic equations), NOT
  standard math/code/logic benchmarks. train.csv (9500 puzzles) is
  the primary training data source, not OpenMathReasoning.
- Correction: Competition prompt automatically appends
  `'\nPlease put your final answer inside \boxed{}...'` to the raw
  puzzle prompt. Uses `apply_chat_template` with single user message.
- Correction: Trainable params with official LoRA config: 880M / 32.5B
  (2.71%). Submission demo baseline scores 0.49 (public).

## User Scenarios & Testing

### User Story 1 - Baseline Evaluation (Priority: P1)

Load Nemotron-3-Nano-30B with `enable_thinking=True`, run inference
on train.csv puzzles using the competition prompt format, score
results with the competition metric, and categorize errors.

**Why this priority**: Cannot improve what you cannot measure. Baseline
is the foundation for all subsequent experiments.

**Independent Test**: Run inference on held-out portion of train.csv,
produce accuracy score and error breakdown by puzzle type (bit
manipulation, encryption, algebraic). Deliverable: error analysis
notebook.

**Acceptance Scenarios**:

1. **Given** Nemotron-3-Nano-30B loaded on G4 VM with thinking enabled,
   **When** inference runs on train.csv validation split, **Then**
   accuracy is recorded with per-puzzle-type breakdown
2. **Given** baseline results, **When** error analysis runs, **Then**
   errors are categorized: wrong answer, wrong binary format, no
   \boxed{}, truncation, empty response

---

### User Story 2 - Prompt Engineering (Priority: P2)

Develop and benchmark prompt strategies. Note: the competition uses
temperature=0.0 (greedy decoding) with `enable_thinking=True`, so
prompt design directly determines the reasoning trace and output.
The competition already appends \boxed{} instructions.

**Why this priority**: Prompt engineering is zero-cost relative to
fine-tuning and may yield significant gains without training.

**Independent Test**: Compare prompt variants on local validation
split; record accuracy delta vs. baseline.

**Acceptance Scenarios**:

1. **Given** baseline (no system prompt), **When** system prompt with
   puzzle-solving guidance applied, **Then** accuracy delta is measured
2. **Given** few-shot examples from train.csv, **When** applied per
   puzzle type, **Then** per-type accuracy is compared to baseline

---

### User Story 3 - Data Curation & Fine-Tuning (Priority: P3)

Fine-tune on the 9500 train.csv puzzle-answer pairs as the primary
dataset. Optionally augment with OpenMathReasoning CoT data for
general reasoning capacity. Use QLoRA (rank <= 32) with the official
target modules. Training data must include `<think>...</think>`
reasoning traces and \boxed{} final answers.

**Why this priority**: Fine-tuning requires validated baseline and
prompt strategy as starting points.

**Independent Test**: Fine-tuned LoRA adapter scores higher than best
prompt-only variant on local validation split.

**Acceptance Scenarios**:

1. **Given** train.csv formatted with thinking traces, **When** QLoRA
   fine-tuning (rank<=32, target: in_proj|out_proj|up_proj|down_proj)
   completes on G4 VM, **Then** training finishes within 12h with
   logged metrics
2. **Given** fine-tuned adapter, **When** evaluated on local split
   with `enable_thinking=True`, **Then** accuracy exceeds prompt-only
   baseline
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
   competition params and `enable_thinking=True`, **Then** model
   generates valid \boxed{} answers matching expected format

### Edge Cases

- Model exceeds VRAM during inference with long sequences (8192 tokens)
- Binary answers must be exact (no leading-zero tolerance in metric)
- LoRA adapter exceeds rank 32 limit
- Model fails to produce \boxed{} answers consistently
- Thinking traces consume most of the 7680 max_tokens budget, leaving
  insufficient room for the final answer
- Train.csv puzzle formats vary across types (bit manipulation vs
  encryption vs algebraic) — model must generalize

## Requirements

### Functional Requirements

- **FR-001**: System MUST load Nemotron-3-Nano-30B and run inference
  with `enable_thinking=True` on competition benchmark
- **FR-002**: System MUST produce accuracy scores and error
  categorization by puzzle type (bit manipulation, encryption,
  algebraic)
- **FR-003**: System MUST support configurable prompt templates
- **FR-004**: System MUST fine-tune with QLoRA (4-bit base + bf16
  LoRA, rank <= 32, target modules: `in_proj|out_proj|up_proj|
  down_proj`) within G4 VM memory constraints
- **FR-005**: System MUST produce answers in \boxed{} LaTeX format
- **FR-006**: System MUST package LoRA adapter as submission.zip with
  adapter_config.json
- **FR-007**: All experiment configs, seeds, and results MUST be
  logged per constitution
- **FR-008**: System MUST evaluate accuracy using competition metric:
  \boxed{} extraction, binary strict match, numeric tolerance 1e-2,
  case-insensitive string fallback
- **FR-009**: Training data MUST include `<think>...</think>` reasoning
  traces matching the model's thinking mode format
- **FR-010**: System MUST use train.csv (9500 puzzles) as primary
  training data source

### Key Entities

- **Experiment**: A single run with config, seed, model, and results
- **Prompt Template**: System prompt + optional few-shot examples
  (competition appends \boxed{} instructions automatically)
- **Training Dataset**: train.csv puzzles (primary) + optional
  OpenMathReasoning CoT augmentation
- **LoRA Adapter**: Fine-tuned adapter weights + adapter_config.json
  (target modules: in_proj, out_proj, up_proj, down_proj)
- **Submission**: Packaged submission.zip for Kaggle

## Success Criteria

### Measurable Outcomes

- **SC-001**: Baseline accuracy established on train.csv validation
  split (compare to submission demo baseline of 0.49)
- **SC-002**: Prompt engineering improves accuracy by measurable delta
  over baseline
- **SC-003**: Fine-tuned LoRA adapter exceeds prompt-only best on
  local split
- **SC-004**: submission.zip passes local validation with vLLM
- **SC-005**: All experiments are reproducible from logged configs
- **SC-006**: Target competitive accuracy >= 0.80 (top 15 on
  leaderboard is ~0.84 as of 2026-04-16)

## Assumptions

- Kaggle G4 VM with RTX PRO 6000 is available for all runs
- Nemotron-3-Nano-30B (30B MoE, 3.5B active) fits in GPU memory for
  inference (~60GB bf16) and QLoRA training (~24GB) on 96GB VRAM
- train.csv (9500 samples, CC-BY-4.0) is sufficient as primary
  training data for domain-specific fine-tuning
- OpenMathReasoning (CC-BY-4.0) is useful for general reasoning
  augmentation but secondary to train.csv
- Competition allows any training framework and approach
- vLLM supports loading LoRA adapters for Nemotron-3-Nano-30B
- RL (GRPO/DPO) is a valid follow-up to SFT but not required for
  initial competitive submission
- The model's `enable_thinking=True` mode produces `<think>...</think>`
  traces that can be trained on

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
| enable_thinking | True |

## Competition Metric Details

### Answer Extraction (`extract_final_answer`)
1. `\boxed{...}` — all matches, return last non-empty
2. Fallback: "The final answer is:" patterns
3. Last numeric value in text
4. Last non-empty line
5. `'NOT_FOUND'` if None

### Answer Verification (`verify`)
- Binary strings (`[01]+`): strict case-insensitive (no leading zeros)
- Numeric: `math.isclose(rel_tol=1e-2, abs_tol=1e-5)`
- Otherwise: case-insensitive string comparison

### Prompt Construction (by competition)
```python
user_content = (
    item.prompt
    + '\nPlease put your final answer inside `\\boxed{}`. '
    + 'For example: `\\boxed{your answer}`'
)
prompt = tokenizer.apply_chat_template(
    [{'role': 'user', 'content': user_content}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
```

## LoRA Configuration (from official submission demo)

```python
LoraConfig(
    r=32,                    # max 32
    lora_alpha=16,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

Trainable: 880,138,240 / 32,458,075,584 (2.71%)
