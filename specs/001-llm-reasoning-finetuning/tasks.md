# Tasks: LLM Reasoning Fine-Tuning Pipeline

**Input**: Design documents from `/specs/001-llm-reasoning-finetuning/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1-US5)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization, directory structure, and shared utilities

- [ ] T001 Create project directory structure per plan.md (src/, experiments/, prompts/, checkpoints/, notebooks/, submissions/, data/splits/)
- [ ] T002 Create requirements.txt with pinned dependencies (torch>=2.2.0, transformers>=4.45.0, peft>=0.12.0, trl>=0.12.0, vllm>=0.12.0, datasets>=3.0.0, accelerate>=1.0.0, bitsandbytes>=0.44.0)
- [ ] T003 [P] Implement seed management utility in src/utils/seeds.py (set PyTorch, NumPy, Python random seeds deterministically)
- [ ] T004 [P] Implement config loading and SHA256 hashing utility in src/utils/config.py (load YAML configs, compute config_hash)
- [ ] T005 [P] Implement experiment CSV logging utility in src/utils/logging.py (append-only experiment_log.csv with fields from data-model.md ExperimentResult)
- [ ] T006 Create initial experiment log CSV header in experiments/experiment_log.csv

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story

**CRITICAL**: No user story work can begin until this phase is complete

- [ ] T007 Implement competition metric (\boxed{} extraction + exact match / numerical tolerance 1e-4) in src/metrics/competition.py
- [ ] T008 Implement prompt template loader in src/prompts/templates.py (load YAML templates, render system prompt + few-shot examples)
- [ ] T009 Create baseline prompt template (no CoT, minimal instruction) in prompts/baseline.yaml
- [ ] T010 Implement validation split creation script in src/data/splits.py (5% holdout from OpenMathReasoning, stratified by difficulty, deterministic seed)
- [ ] T011 Implement evaluation pipeline in src/evaluate.py (load model via vLLM with competition params: temp=0.0, top_p=1.0, max_tokens=7680, max_model_len=8192; run inference on split; score with competition metric; output JSON results)
- [ ] T012 Create first experiment config YAML template in experiments/configs/exp-001-baseline.yaml (model_name, seed=42, competition eval params, prompt_template=baseline.yaml)

**Checkpoint**: Foundation ready — user story implementation can now begin

---

## Phase 3: User Story 1 — Baseline Evaluation (Priority: P1) MVP

**Goal**: Establish measurable baseline performance with unmodified 30B model

**Independent Test**: Run inference on local validation split, produce accuracy score and per-domain error breakdown

- [ ] T013 [US1] Run src/data/splits.py to create frozen validation split at data/splits/validation.jsonl (5% OpenMathReasoning CoT, stratified by difficulty, seed=42)
- [ ] T014 [US1] Run src/evaluate.py with exp-001-baseline.yaml config against data/splits/validation.jsonl — record overall and per-domain accuracy in experiments/results/exp-001-baseline.json
- [ ] T015 [US1] Log exp-001-baseline results to experiments/experiment_log.csv (run_id, config_hash, accuracy, boxed_rate, truncation_rate, peak_vram_gb, decision=adopt as baseline)
- [ ] T016 [US1] Build error analysis notebook at notebooks/error_analysis.ipynb (categorize errors: wrong answer, format error/no \boxed{}, truncation/hit token limit, empty response; show per-domain breakdown with examples)

**Checkpoint**: Baseline accuracy established. All subsequent experiments compare against this.

---

## Phase 4: User Story 2 — Prompt Engineering (Priority: P2)

**Goal**: Maximize accuracy through prompt optimization alone (greedy decoding — prompt directly determines output)

**Independent Test**: Compare prompt variant accuracy deltas vs. baseline on local validation split

- [ ] T017 [US2] Create CoT system prompt template in prompts/cot-boxed.yaml (chain-of-thought reasoning with explicit \boxed{} answer instruction)
- [ ] T018 [US2] Create experiment config experiments/configs/exp-002-cot-prompt.yaml (treatment_variable=system_prompt, control_run_id=exp-001-baseline)
- [ ] T019 [US2] Run evaluation with exp-002-cot-prompt.yaml, log results to experiments/results/exp-002-cot-prompt.json and experiment_log.csv
- [ ] T020 [P] [US2] Create math few-shot examples prompt template in prompts/cot-math-fewshot.yaml (algebraic/geometric reasoning ending in \boxed{})
- [ ] T021 [P] [US2] Create code few-shot examples prompt template in prompts/cot-code-fewshot.yaml (step-by-step algorithm tracing with numeric \boxed{} answer)
- [ ] T022 [US2] Create experiment config experiments/configs/exp-003-math-fewshot.yaml (treatment_variable=math_few_shot, control_run_id=exp-002-cot-prompt)
- [ ] T023 [US2] Run evaluation with exp-003-math-fewshot.yaml, log results
- [ ] T024 [US2] Create experiment config experiments/configs/exp-004-code-fewshot.yaml (treatment_variable=code_few_shot, control_run_id=exp-002-cot-prompt)
- [ ] T025 [US2] Run evaluation with exp-004-code-fewshot.yaml, log results
- [ ] T026 [US2] Create combined best-prompt template in prompts/best-prompt.yaml (merge winning elements from exp-002 through exp-004)
- [ ] T027 [US2] Create experiment config experiments/configs/exp-005-combined-prompt.yaml and run evaluation, log results
- [ ] T028 [US2] Record adopt/revert decision for each prompt experiment; document best prompt template ID in experiment log notes

**Checkpoint**: Best prompt template identified. This becomes the format for fine-tuning data preparation.

---

## Phase 5: User Story 3 — Data Curation & Fine-Tuning (Priority: P3)

**Goal**: Improve reasoning through QLoRA supervised fine-tuning on curated CoT data

**Independent Test**: Fine-tuned LoRA adapter scores higher than best prompt-only variant on local validation split

- [ ] T029 [US3] Implement OpenMathReasoning data curation pipeline in src/data/prepare.py (download CoT solutions, filter: correct answers only, remove repetitive patterns, ensure \boxed{} format, remove >7680 tokens, balance domains, prefer shorter solutions)
- [ ] T030 [US3] Run src/data/prepare.py to produce filtered training dataset; log dataset metadata (dataset_id, num_samples, license=CC-BY-4.0, filtering_method, domain_distribution) per data-model.md TrainingDataset entity
- [ ] T031 [US3] Implement QLoRA fine-tuning script in src/train.py (load 30B model in 4-bit via BitsAndBytesConfig, apply LoRA via PEFT with rank<=32, train with TRL SFTTrainer, save adapter_config.json + weights to checkpoints/)
- [ ] T032 [US3] Create experiment config experiments/configs/exp-010-qlora-r16.yaml (lora_rank=16, lora_alpha=32, lr=2e-4, batch_size=4, grad_accum=8, max_seq_length=2048, epochs=1, seed=42, treatment_variable=qlora_r16, control_run_id=exp-005-combined-prompt)
- [ ] T033 [US3] Run QLoRA training with exp-010-qlora-r16.yaml, save adapter to checkpoints/exp-010-qlora-r16/
- [ ] T034 [US3] Evaluate exp-010 adapter with src/evaluate.py (vLLM + LoRA loading, competition params), log results
- [ ] T035 [US3] Create experiment config experiments/configs/exp-011-qlora-r32.yaml (treatment_variable=lora_rank, lora_rank=32, control_run_id=exp-010-qlora-r16)
- [ ] T036 [US3] Run QLoRA training with exp-011-qlora-r32.yaml, save adapter to checkpoints/exp-011-qlora-r32/
- [ ] T037 [US3] Evaluate exp-011 adapter, log results, compare to exp-010 and prompt-only baseline
- [ ] T038 [US3] Create experiment config experiments/configs/exp-012-lr-sweep.yaml (treatment_variable=learning_rate, use best rank from exp-010/011, try lr=1e-4, control_run_id=best of exp-010/011)
- [ ] T039 [US3] Run QLoRA training with exp-012, evaluate, log results
- [ ] T040 [US3] Record adopt/revert decisions for all SFT experiments; copy best adapter to checkpoints/best-sft/

**Checkpoint**: Best SFT adapter identified and saved. Accuracy exceeds prompt-only baseline.

---

## Phase 6: User Story 4 — RL Ablation (Priority: P4, Optional)

**Goal**: Test whether RL post-training (GRPO/DPO) improves the SFT adapter

**Independent Test**: RL-enhanced adapter scores higher than best SFT adapter on local validation split

- [ ] T041 [US4] Implement RL training script (GRPO or DPO) in src/train_rl.py (load best SFT adapter, apply RL using correct/incorrect answer pairs as reward, save enhanced adapter)
- [ ] T042 [US4] Create experiment config experiments/configs/exp-020-grpo.yaml (treatment_variable=grpo_rl, control_run_id=best-sft, reward=correct_answer_match)
- [ ] T043 [US4] Run RL training with exp-020-grpo.yaml, save adapter to checkpoints/exp-020-grpo/
- [ ] T044 [US4] Evaluate exp-020 adapter with competition metric, log results, compare to best SFT
- [ ] T045 [US4] Record adopt/revert decision; if adopted, copy to checkpoints/best/; if reverted, copy best-sft to checkpoints/best/

**Checkpoint**: Best overall adapter (SFT or SFT+RL) saved to checkpoints/best/

---

## Phase 7: User Story 5 — Submission Pipeline (Priority: P5)

**Goal**: Package best adapter and produce competition-compliant submission

**Independent Test**: submission.zip loads with vLLM, produces valid \boxed{} answers, adapter_config.json has rank <= 32

- [ ] T046 [US5] Implement adapter packaging script in src/package.py (copy adapter_config.json + adapter_model.safetensors from checkpoints/best/ into submission.zip; validate rank<=32 in config; validate base_model field)
- [ ] T047 [US5] Run src/package.py to produce submissions/submission.zip
- [ ] T048 [US5] Validate submission.zip locally: unzip, load adapter with vLLM using competition params, run on validation split subset (50 samples), confirm \boxed{} output and accuracy matches expectations
- [ ] T049 [US5] Create public documentation notebook at notebooks/submission_demo.ipynb (document methods, datasets, techniques used — required for prize eligibility)
- [ ] T050 [US5] Submit submission.zip to Kaggle competition

**Checkpoint**: Submission uploaded to Kaggle. Public documentation published.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements across all user stories

- [ ] T051 [P] Update experiments/experiment_log.csv with final summary row documenting best configuration path (baseline → prompt → SFT → RL decision)
- [ ] T052 [P] Review all experiment configs in experiments/configs/ for completeness (every run has config_hash, seed, treatment_variable, control_run_id, decision)
- [ ] T053 Verify reproducibility: re-run best experiment from committed config with same seed, confirm identical results
- [ ] T054 [P] Update error_analysis.ipynb with final model comparison (baseline vs. prompt vs. SFT vs. best)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Setup — BLOCKS all user stories
- **US1 Baseline (Phase 3)**: Depends on Foundational
- **US2 Prompt Engineering (Phase 4)**: Depends on US1 (needs baseline as control)
- **US3 Fine-Tuning (Phase 5)**: Depends on US2 (needs best prompt as training format)
- **US4 RL Ablation (Phase 6)**: Depends on US3 (needs best SFT adapter). Optional — skip if time-constrained
- **US5 Submission (Phase 7)**: Depends on US3 or US4 (needs best adapter in checkpoints/best/)
- **Polish (Phase 8)**: Depends on all prior phases

### Within Each User Story

- Configs created before training/evaluation runs
- Training before evaluation (for fine-tuning stories)
- Results logged before decisions recorded
- Decisions recorded before next experiment begins

### Parallel Opportunities

**Phase 1 (Setup)**:
```text
Parallel: T003, T004, T005 (independent utility modules)
```

**Phase 4 (Prompt Engineering)**:
```text
Parallel: T020, T021 (independent prompt template files)
```

**Phase 8 (Polish)**:
```text
Parallel: T051, T052, T054 (independent documentation tasks)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: US1 Baseline Evaluation
4. **STOP and VALIDATE**: Baseline accuracy established, error analysis complete
5. Submit baseline (no adapter) to Kaggle to verify submission pipeline

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. US1 Baseline → Measurable starting point (MVP!)
3. US2 Prompt Engineering → Zero-cost accuracy improvement
4. US3 Fine-Tuning → LoRA adapter with validated improvement
5. US4 RL Ablation → Optional post-training optimization
6. US5 Submission → Competition-ready submission.zip
7. Each story adds value and is independently verifiable

### Critical Path

```text
T001-T006 → T007-T012 → T013-T016 → T017-T028 → T029-T040 → T046-T050
(Setup)     (Foundation) (Baseline)  (Prompts)    (Fine-Tune)  (Submit)
                                                       ↓
                                               T041-T045 (RL, optional)
```

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story is independently completable and verifiable
- Constitution requires: one variable changed per experiment (ablation discipline)
- All evaluation MUST use competition params (temp=0, top_p=1, max_tokens=7680)
- Commit config + results after each experiment per constitution
- LoRA rank MUST be <= 32 in all training configs
