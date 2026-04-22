# Tasks: LLM Reasoning Fine-Tuning Pipeline

**Input**: Design documents from `/specs/001-llm-reasoning-finetuning/`
**Prerequisites**: plan.md (rev 3), spec.md (clarified 2026-04-21), research.md, data-model.md, contracts/
**Revision**: 4 (Colab Pro pipeline — aligned to revised plan)
**Primary Platform**: Colab Pro L4 (24 GB VRAM)

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1-US4)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization, directory structure, shared utilities

- [X] T001 Create project directory structure per plan.md (src/solvers/, src/trace_generators/, src/data/, src/metrics/, src/utils/, experiments/configs/, checkpoints/, submissions/, notebooks/)
- [X] T002 Create requirements.txt with pinned dependencies (torch>=2.2, transformers>=4.45, peft>=0.12, trl>=0.12, vllm>=0.12, datasets>=3.0, accelerate>=1.0, bitsandbytes>=0.44, polars>=1.0, pyyaml>=6.0)
- [X] T003 [P] Implement seed management in src/utils/seeds.py
- [X] T004 [P] Implement config loading and SHA256 hashing in src/utils/config.py
- [X] T005 [P] Implement experiment CSV logging in src/utils/logging.py

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story

- [X] T006 Implement competition metric (extract_final_answer + verify) in src/metrics/competition.py
- [X] T007 Implement puzzle classifier (train.csv → 9 subcategories) in src/data/prepare.py
- [X] T008 Implement stratified train/val split (90/10) in src/data/splits.py
- [X] T009 Implement abstract solver interface in src/solvers/base.py
- [X] T010 Implement abstract trace generator interface in src/trace_generators/base.py
- [X] T011 Implement SFT formatter (ChatML + `<think>` + `\boxed{}`, validation gates) in src/data/format_sft.py
- [X] T012 Implement vLLM evaluation pipeline in src/evaluate.py
- [X] T013 Implement adapter packaging + validation in src/package.py
- [X] T014 Implement QLoRA training with PEFT/TRL SFTTrainer in src/train.py

**Checkpoint**: Foundation complete — all modules implemented

---

## Phase 3: User Story 1 — Baseline Evaluation (Priority: P1) MVP

**Goal**: Establish measurable baseline accuracy of unmodified model

**Independent Test**: `python -m src.evaluate --split data/splits/val_ids.json` produces ~0.49

- [X] T015 [US1] Create baseline experiment config in experiments/configs/exp-001-baseline.yaml
- [ ] T016 [US1] Run baseline evaluation on validation split, save results to experiments/results/exp-001-baseline.json
- [ ] T017 [US1] Log baseline to experiments/experiment_log.csv

**Checkpoint**: Baseline accuracy established (~0.49)

---

## Phase 4: User Story 2 — Solvers & Traces (Priority: P2)

**Goal**: Write solvers + trace generators for all 7 categories. Easy categories target 100%, hard categories target best achievable.

**Independent Test**: Each solver verified on train.csv. Traces formatted as SFT data.

### 4a. Easy Category Solvers (100% solvable — 66.8% floor)

- [X] T018 [P] [US2] Implement numeral solver in src/solvers/numeral.py
- [X] T019 [P] [US2] Implement gravity solver in src/solvers/gravity.py
- [X] T020 [P] [US2] Implement unit conversion solver in src/solvers/unit_conversion.py
- [X] T021 [P] [US2] Implement cipher solver in src/solvers/cipher.py

### 4b. Easy Category Trace Generators

- [X] T022 [P] [US2] Implement numeral trace generator in src/trace_generators/numeral_traces.py
- [X] T023 [P] [US2] Implement gravity trace generator in src/trace_generators/gravity_traces.py
- [X] T024 [P] [US2] Implement unit conversion trace generator in src/trace_generators/unit_conversion_traces.py
- [X] T025 [P] [US2] Implement cipher trace generator in src/trace_generators/cipher_traces.py

### 4c. Hard Category Solvers

- [X] T026 [P] [US2] Implement bit manipulation solver (52 gate types, 5 levels) in src/solvers/bit_manipulation.py
- [X] T027 [P] [US2] Implement equation solver (4 transforms × 32 operators) in src/solvers/equation.py
- [X] T028 [P] [US2] Implement cryptarithm solver (concatenation detection, carry-free, brute-force) in src/solvers/cryptarithm.py

### 4d. Hard Category Trace Generators

- [X] T029 [P] [US2] Implement bit manipulation trace generator (bit-serial) in src/trace_generators/bit_manipulation_traces.py
- [X] T030 [P] [US2] Implement equation trace generator in src/trace_generators/equation_traces.py
- [X] T031 [P] [US2] Implement cryptarithm trace generator in src/trace_generators/cryptarithm_traces.py

**Checkpoint**: All 7 solver + trace generator pairs implemented

---

## Phase 5: User Story 3 — Colab Pipeline & Full SFT (Priority: P3)

**Goal**: Fix data quality issues, build Colab notebook, run full SFT, evaluate

**Independent Test**: Fine-tuned adapter on val split exceeds baseline (>0.49). Notebook runs end-to-end on Colab Pro L4 within 2h.

### 5a. Code Fixes (prerequisite for clean training data)

> From the Colab Pro pipeline plan. These fixes land in `src/` and benefit
> both Kaggle and Colab. Must be done before full SFT to avoid duplicate
> `\boxed{}` in training data and OOM on L4.

- [X] T032 [P] [US3] Drop inline `\boxed{}` from thinking_text in src/trace_generators/numeral_traces.py:184
- [X] T033 [P] [US3] Drop inline `\boxed{}` from thinking_text in src/trace_generators/gravity_traces.py:167
- [X] T034 [P] [US3] Drop inline `\boxed{}` from thinking_text in src/trace_generators/cipher_traces.py:191
- [X] T035 [P] [US3] Drop inline `\boxed{}` from thinking_text in src/trace_generators/cryptarithm_traces.py:296
- [X] T036 [P] [US3] Drop inline `\boxed{}` from thinking_text in src/trace_generators/bit_manipulation_traces.py:263
- [X] T037 [US3] Fix numeral CLI output keys in src/trace_generators/numeral_traces.py:251-252 — `"thinking"` → `"thinking_text"`, `"answer"` → `"final_answer"`
- [X] T038 [US3] Lower default --max-tokens from 7680 to 7200 in src/data/format_sft.py:100
- [X] T039 [US3] Add gradient_checkpointing pass-through in src/train.py SFTConfig (~line 172)
- [X] T040 [US3] Add `gradient_checkpointing: true` in experiments/configs/exp-011-full-sft.yaml

### 5b. Review Fixes (from code review audit)

- [X] T041 [US3] Add `trust_remote_code=True` to vLLM LLM() kwargs in src/evaluate.py:80
- [X] T042 [US3] Add `max_lora_rank=32` to vLLM LLM() kwargs when loading adapter in src/evaluate.py:84
- [X] T043 [US3] Fix misleading `overall_accuracy = train_loss` placeholder in src/train.py:353

### 5c. Colab Notebook

- [X] T044 [US3] Create Colab orchestrator notebook at notebooks/colab_pipeline.ipynb (~12 cells, thin wrapper over `python -m src.*` CLIs, no Unsloth, no base64, no HF .generate())
- [X] T045 [US3] Add `assert 'YOUR_USER' not in REPO_URL` guard in colab notebook cell 2
- [X] T046 [US3] Add `assert os.path.exists('data/train.csv')` after Kaggle download in colab notebook cell 3
- [X] T047 [US3] Update colab notebook VRAM figure from 22.5 GB to 24 GB

### 5d. Verification

- [X] T048 [US3] Smoke-test imports on CPU: `python -c "from src.train import load_model_for_training; from src.data.format_sft import format_trace_to_sft"` — verify no ImportError
- [X] T049 [US3] Spot-check SFT example: run `python -m src.data.format_sft --max-tokens 7200` on existing traces, grep first example's assistant content for `\boxed` — must appear exactly once, after `</think>` (RESULT: stale trace files have double boxed; fresh generation confirmed clean — T050 regen required)

### 5e. Generate Data & Run Full SFT

- [X] T050 [US3] Generate all CoT traces for all categories (run solver + trace generator per category), save to data/traces/
- [X] T051 [US3] Run `python -m src.data.format_sft --max-tokens 7200 --output data/sft/train_sft_full.jsonl` — verify validation passes
- [ ] T052 [US3] Run full SFT on Colab Pro L4 via `python -m src.train --config experiments/configs/exp-011-full-sft.yaml` — watch for: collator mask 40-70%, loss finite, adapter files present
- [ ] T053 [US3] Run evaluation via `python -m src.evaluate --adapter checkpoints/exp-011-full-sft --config experiments/configs/exp-011-full-sft.yaml --output experiments/results/exp-011.json` — target >0.49 overall
- [ ] T054 [US3] End-to-end: run colab_pipeline.ipynb top-to-bottom on Colab Pro L4. Expected ~75-100 min total. Verify submission ZIP <200 MB.

**Checkpoint**: Full SFT adapter trained and evaluated. Colab pipeline validated end-to-end.

---

## Phase 6: User Story 4 — Submission & Optimization (Priority: P4)

**Goal**: Package best adapter, submit to Kaggle, optimize if needed

**Independent Test**: submission.zip scores >= 0.85 on public leaderboard

### 6a. Submit

- [ ] T055 [US4] Package adapter via `python -m src.package --adapter checkpoints/exp-011-full-sft --output submissions/exp-011.zip`
- [ ] T056 [US4] Submit to Kaggle via `kaggle competitions submit`, record public score
- [ ] T057 [US4] If accuracy < 0.85: proceed to ablation (T058+). If >= 0.85: skip to Polish (Phase 7).

### 6b. Ablation & Optimization (conditional)

- [ ] T058 [P] [US4] Create experiment config for lr ablation (lr=1e-4) in experiments/configs/exp-012-lr-ablation.yaml
- [ ] T059 [P] [US4] Create experiment config for batch_size=2 (safer L4 fit) in experiments/configs/exp-013-bs-ablation.yaml
- [ ] T060 [US4] Run ablation experiments, evaluate, compare to exp-011
- [ ] T061 [US4] Select best adapter, copy to checkpoints/best/

**Checkpoint**: Competitive submission on Kaggle leaderboard.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final optimization, documentation, deadline submissions

- [ ] T062 [P] Optional: RL (GRPO) ablation on best SFT adapter — experiments/configs/exp-020-grpo.yaml
- [ ] T063 [P] Create public documentation notebook (required for prizes) at notebooks/documentation.ipynb
- [ ] T064 Verify reproducibility: re-run best experiment from config with same seed
- [ ] T065 Select 2 final submissions, submit before June 15, 2026 deadline

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 — BLOCKS all user stories
- **Phase 3 (US1 Baseline)**: Depends on Phase 2
- **Phase 4 (US2 Solvers)**: Depends on Phase 2 (independent of US1)
- **Phase 5 (US3 Colab + SFT)**: Code fixes (5a-5c) independent of Phases 3-4. Data generation (5e) depends on Phase 4 solvers + Phase 5a fixes.
- **Phase 6 (US4 Submission)**: Depends on Phase 5 full SFT
- **Phase 7 (Polish)**: Depends on Phase 6

### Parallel Opportunities

**Phase 2**: T006-T014 touch different files — many can run in parallel

**Phase 4**: All solver pairs are independent:
```
Parallel: T018+T022, T019+T023, T020+T024, T021+T025 (easy)
Parallel: T026+T029, T027+T030, T028+T031 (hard)
```

**Phase 5a**: T032-T036 all touch different trace generator files — fully parallel

**Phase 5b**: T041-T043 touch different files (evaluate.py, train.py) — parallel

**Phase 6b**: T058, T059 are independent config files — parallel

---

## Implementation Strategy

### Colab Pro Path (current focus)

All code is implemented. Remaining work is data generation + training + eval:

1. **Verify fixes** (T048-T049) — CPU smoke tests
2. **Generate data** (T050-T051) — run solvers + traces + format SFT
3. **Train on Colab L4** (T052) — ~25-35 min with gradient checkpointing
4. **Evaluate with vLLM** (T053) — ~15-25 min after freeing training model
5. **End-to-end notebook run** (T054) — full validation
6. **Package + submit** (T055-T056)

### Critical Path

```
T048-T049 → T050-T051 → T052 → T053 → T054 → T055-T056
(verify)    (data gen)   (train) (eval)  (e2e)  (submit)
```

### What NOT to do (lessons from Kaggle)

- No Unsloth — use plain transformers + peft + bitsandbytes
- No base64-embedding of src/ — clone the repo
- No HF .generate() for eval — use src/evaluate.py with vLLM
- No reimplementing train.py/evaluate.py in notebook cells
- No hardcoded hyperparams in notebooks — use YAML configs

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks
- [Story] label maps task to specific user story
- Competition deadline: 2026-06-15
- Primary GPU: Colab Pro L4 (24 GB GDDR6). Kaggle G4 (96 GB) is fallback.
- Training requires gradient_checkpointing=true on L4
- Effective token budget: 7200 (reserves ~480 tokens for prompt overhead vs competition max 7680)
- LoRA rank MUST be <= 32, target_modules: `in_proj|out_proj|up_proj|down_proj`
- lora_alpha=16 with r=32 gives scaling ratio 0.5 (from official demo — consider testing alpha=32 or 64)
- per_device_train_batch_size=4 may OOM on L4 with max_seq_length=7680 — if so, try batch_size=2 (T059)
