# Tasks: LLM Reasoning Fine-Tuning Pipeline

**Input**: Design documents from `/specs/001-llm-reasoning-finetuning/`
**Prerequisites**: plan.md (rev 3), spec.md, deep-research.md, competition-reference.md
**Revision**: 3 (solver-first architecture)

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1-US4)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization, directory structure, shared utilities

- [X] T001 Create project directory structure per plan.md (src/solvers/, src/trace_generators/, src/data/, src/metrics/, src/utils/, experiments/configs/, experiments/results/, checkpoints/, submissions/, notebooks/, data/splits/, data/traces/, data/sft/)
- [X] T002 Create requirements.txt with pinned dependencies (torch>=2.2.0, transformers>=4.45.0, peft>=0.12.0, trl>=0.12.0, vllm>=0.12.0, datasets>=3.0.0, accelerate>=1.0.0, bitsandbytes>=0.44.0, polars>=1.0.0)
- [X] T003 [P] Implement seed management utility in src/utils/seeds.py (set PyTorch, NumPy, Python random seeds deterministically)
- [X] T004 [P] Implement config loading and SHA256 hashing utility in src/utils/config.py (load YAML configs, compute config_hash)
- [X] T005 [P] Implement experiment CSV logging utility in src/utils/logging.py (append-only experiment_log.csv with fields: run_id, config_hash, timestamp, hypothesis, treatment_variable, control_run_id, seed, categories_included, overall_accuracy, per_category_accuracy, decision)
- [X] T006 Create initial experiment log CSV header in experiments/experiment_log.csv
- [X] T007 Download competition train.csv to data/train.csv (from Kaggle competition data)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story

**CRITICAL**: No user story work can begin until this phase is complete

- [X] T008 Implement exact competition metric in src/metrics/competition.py: `extract_final_answer()` (boxed extraction with fallback patterns) and `verify()` (binary strict match, numeric rel_tol=1e-2, string case-insensitive) — copy logic exactly from competition-reference.md
- [X] T009 Implement puzzle category classifier in src/data/prepare.py (load train.csv, classify each row into: numeral, gravity, unit_conversion, cipher, bit_manipulation, equation_numeric_deduce, equation_numeric_guess, cryptarithm_deduce, cryptarithm_guess; output data/puzzles_classified.jsonl)
- [X] T010 Implement train/validation split in src/data/splits.py (90/10 stratified by category, deterministic seed=42, output data/splits/train_ids.json and val_ids.json)
- [X] T011 Implement abstract solver interface in src/solvers/base.py (BaseSolver with solve() and verify() methods per contracts/experiment-contract.md)
- [X] T012 Implement abstract trace generator interface in src/trace_generators/base.py (BaseTraceGenerator with generate_trace() returning thinking_text + final_answer, token counting, 7680 limit check)
- [X] T013 Implement SFT data formatter in src/data/format_sft.py (load traces, format as ChatML messages with `<think>...</think>` + `\boxed{}`, verify token count < 7680, output data/sft/train_sft.jsonl)
- [X] T014 Implement evaluation pipeline in src/evaluate.py (load model via vLLM with competition params: temp=0.0, top_p=1.0, max_tokens=7680, max_model_len=8192, enable_thinking=True; run inference; score with competition metric; report per-category accuracy)
- [X] T015 Implement adapter packaging in src/package.py (copy adapter_config.json + adapter_model.safetensors into submission.zip; validate rank<=32 and target_modules match regex)

**Checkpoint**: Foundation ready — user story implementation can now begin

---

## Phase 3: User Story 1 — Baseline Evaluation (Priority: P1) MVP

**Goal**: Establish measurable baseline accuracy of unmodified 30B model on competition puzzles

**Independent Test**: Run inference on validation split, produce per-category accuracy breakdown. Compare to submission demo baseline of 0.49.

- [X] T016 [US1] Run src/data/prepare.py to classify all 9500 train.csv puzzles into categories, output data/puzzles_classified.jsonl
- [X] T017 [US1] Run src/data/splits.py to create frozen validation split at data/splits/ (10% holdout, stratified, seed=42)
- [X] T018 [US1] Create baseline experiment config in experiments/configs/exp-001-baseline.yaml (model_name, seed=42, no adapter, competition eval params, prompt_template=none)
- [ ] T019 [US1] Run src/evaluate.py with exp-001-baseline.yaml against validation split — record overall and per-category accuracy in experiments/results/exp-001-baseline.json
- [ ] T020 [US1] Log exp-001-baseline results to experiments/experiment_log.csv (decision=adopt as baseline)
- [ ] T021 [US1] Build error analysis notebook at notebooks/error_analysis.ipynb (per-category accuracy, error types: wrong answer, wrong binary format, no \boxed{}, truncation, empty response; sample errors per category)

**Checkpoint**: Baseline accuracy established (~0.49 expected). All subsequent experiments compare against this.

---

## Phase 4: User Story 2 — Easy Category Solvers & Traces (Priority: P2)

**Goal**: Write solvers and CoT trace generators for 4 easy categories achieving 100% solve rate = 66.8% accuracy floor

**Independent Test**: Each solver verified 100% correct on train.csv for its category. CoT traces formatted as SFT data.

### 4a. Roman Numeral (1576 samples, target: 100%)

- [X] T022 [P] [US2] Implement Roman numeral solver in src/solvers/numeral.py (enumerate all 1-100 Roman numerals, parse prompt examples, extract target, convert)
- [X] T023 [P] [US2] Implement numeral trace generator in src/trace_generators/numeral_traces.py (step-by-step decomposition: thousands → hundreds → tens → ones → concatenation → \boxed{})
- [X] T024 [US2] Verify numeral solver: 1576/1576 = 100%: run on all 1576 numeral puzzles in train.csv, assert 100% accuracy with competition metric

### 4b. Gravity (1597 samples, target: 100%)

- [X] T025 [P] [US2] Implement gravity solver in src/solvers/gravity.py (parse examples to extract (t,d) pairs, derive rate=d/t^2, apply to target, format X.XX)
- [X] T026 [P] [US2] Implement gravity trace generator in src/trace_generators/gravity_traces.py (rate-first decomposition, multi-step arithmetic, rate consistency verification against EX2, format to X.XX)
- [X] T027 [US2] Verify gravity solver: 1597/1597 = 100%: run on all 1597 gravity puzzles, assert 100% accuracy

### 4c. Unit Conversion (1594 samples, target: 100%)

- [X] T028 [P] [US2] Implement unit conversion solver in src/solvers/unit_conversion.py (derive factor=out/in from examples, apply factor*target, format X.XX)
- [X] T029 [P] [US2] Implement unit conversion trace generator in src/trace_generators/unit_conversion_traces.py (rate derivation, multiplication steps, rate consistency check, format X.XX)
- [X] T030 [US2] Verify unit conversion solver: 1594/1594 = 100%: run on all 1594 puzzles, assert 100% accuracy

### 4d. Cipher (1576 samples, target: 100%)

- [X] T031 [P] [US2] Implement cipher solver in src/solvers/cipher.py (extract char mappings from example pairs, handle unmapped chars via vocabulary fill from ~90 Wonderland words)
- [X] T032 [P] [US2] Implement cipher trace generator in src/trace_generators/cipher_traces.py (build mapping table, char-by-char decryption, vocabulary matching for gaps, verify decryption)
- [X] T033 [US2] Verify cipher solver: 1576/1576 = 100%: run on all 1576 cipher puzzles, assert 100% accuracy

### Integration

- [ ] T034 [US2] Generate all CoT traces for easy categories, save to data/traces/ (numeral_traces.jsonl, gravity_traces.jsonl, unit_conversion_traces.jsonl, cipher_traces.jsonl)
- [ ] T035 [US2] Verify all traces fit within 7680 token limit using tokenizer token counting
- [ ] T036 [US2] Run src/data/format_sft.py on easy category traces to produce data/sft/train_sft_easy.jsonl

**Checkpoint**: 4 solvers verified 100% on train.csv. SFT data ready for 6343 puzzles.

---

## Phase 5: User Story 3 — First SFT & Hard Solvers (Priority: P3)

**Goal**: Train on easy traces, verify model learns, then add hard categories for ~0.85

**Independent Test**: Fine-tuned adapter exceeds baseline on validation split. Hard category solvers verified on train.csv.

### 5a. First SFT on Easy Categories

- [X] T037 [US3] Create experiment config experiments/configs/exp-010-easy-sft.yaml (QLoRA r=32, lora_alpha=16, target_modules=`r".*\.(in_proj|out_proj|up_proj|down_proj)$"`, lr=2e-4, batch_size=4, grad_accum=8, max_seq_length=4096, epochs=1, seed=42, categories=numeral+gravity+unit_conversion+cipher)
- [X] T038 [US3] Implement QLoRA training script in src/train.py (load 30B model in 4-bit via BitsAndBytesConfig nf4, apply LoRA via PEFT with config from YAML, train with TRL SFTTrainer on data/sft/train_sft_easy.jsonl, save adapter to checkpoints/)
- [ ] T039 [US3] Run QLoRA training with exp-010-easy-sft.yaml, save adapter to checkpoints/exp-010-easy-sft/
- [ ] T040 [US3] Evaluate exp-010 adapter on validation split with src/evaluate.py, log results, verify near-100% on easy categories (~0.67 overall)
- [ ] T041 [US3] Package exp-010 adapter as submissions/submission-easy.zip, submit to Kaggle for first score

### 5b. Bit Manipulation Solver (1602 samples, target: 85%)

- [X] T042 [P] [US3] Implement bit manipulation solver in src/solvers/bit_manipulation.py (per-bit boolean function search through 52 gate types: Level 0 constants → Level 1 identity/NOT → Level 2 AND/OR/XOR/NAND/NOR/XNOR+4 negation variants → Level 3 MAJ/CHO/PAR3/AO/OA/AX/OX/XA/XO → Level 4 AOA/OAO/PAR4/XX/AXA; verify candidate against test input)
- [X] T043 [P] [US3] Implement bit manipulation trace generator in src/trace_generators/bit_manipulation_traces.py (bit-serial gate computation: spell out each operation one bit at a time like `0&1=0 1&1=1`; include verification step)
- [X] T044 [US3] Verify bit manipulation solver on train.csv: 960/1602 = 60% (target 85%, needs improvement)

### 5c. Equation Solver (732 samples, target: 76-90%)

- [X] T045 [P] [US3] Implement equation solver in src/solvers/equation.py (4 operand transforms: AB_CD, BA_DC, AB_CD→YX, BA_DC→YX × 32 operators; frequency-ordered brute force scan; EX2 verification to catch coincidental matches)
- [X] T046 [P] [US3] Implement equation trace generator in src/trace_generators/equation_traces.py (parse → scan → lock → apply → answer format)
- [X] T047 [US3] Verify equation solver on train.csv: 309/687 = 45% (target 76-90%, needs more operations)

### 5d. Cryptarithm Solver (823 samples, target: ~8%)

- [X] T048 [P] [US3] Implement cryptarithm solver in src/solvers/cryptarithm.py (detect concatenation/reverse concatenation as baseline; accept low solve rate)
- [X] T049 [P] [US3] Implement cryptarithm trace generator in src/trace_generators/cryptarithm_traces.py (traces for solvable subset only)
- [X] T050 [US3] Verify cryptarithm solver on train.csv: 0/868 = 0% (puzzle format is symbol transformations, not traditional cryptarithm)

### 5e. Full SFT with All Categories

- [ ] T051 [US3] Generate all CoT traces for hard categories, save to data/traces/ (bit_manipulation_traces.jsonl, equation_traces.jsonl, cryptarithm_traces.jsonl)
- [ ] T052 [US3] Run src/data/format_sft.py on all category traces to produce data/sft/train_sft_full.jsonl
- [X] T053 [US3] Create experiment config experiments/configs/exp-011-full-sft.yaml (same QLoRA config, categories=all, treatment_variable=add_hard_categories, control_run_id=exp-010)
- [ ] T054 [US3] Run QLoRA training with exp-011-full-sft.yaml, save adapter to checkpoints/exp-011-full-sft/
- [ ] T055 [US3] Evaluate exp-011 adapter on validation split, log per-category accuracy, compare to exp-010 (~0.85 target)

**Checkpoint**: Best SFT adapter with all categories. Per-category accuracy validated.

---

## Phase 6: User Story 4 — Submission & Optimization (Priority: P4)

**Goal**: Optimize adapter, submit best version to Kaggle

**Independent Test**: submission.zip scores >= 0.85 on public leaderboard

### 6a. Ablation & Optimization

- [ ] T056 [US4] Inspect minimum logprob per trace from exp-011 training — identify weak spots (traces where model is least confident)
- [ ] T057 [US4] Create experiment config experiments/configs/exp-012-trace-refinement.yaml (refine traces for categories with <100% accuracy, treatment_variable=trace_quality)
- [ ] T058 [US4] Refine trace generators where model struggles (tokenization issues, arithmetic steps too complex, bit-serial ambiguities), regenerate traces, retrain
- [ ] T059 [US4] Create experiment config experiments/configs/exp-013-lr-ablation.yaml (treatment_variable=learning_rate, try lr=1e-4 vs 2e-4)
- [ ] T060 [US4] Run exp-013 training, evaluate, compare to exp-011
- [ ] T061 [US4] Select best adapter based on validation accuracy, copy to checkpoints/best/

### 6b. Submission

- [ ] T062 [US4] Package best adapter as submissions/submission.zip using src/package.py (verify adapter_config.json has rank<=32 and correct target_modules)
- [ ] T063 [US4] Submit submission.zip to Kaggle, record public score
- [ ] T064 [US4] If accuracy < 0.85: iterate on trace quality for weakest categories; if >= 0.85: proceed to polish

**Checkpoint**: Competitive submission on Kaggle leaderboard.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final optimization, documentation, second submission

- [ ] T065 [P] Try RL (GRPO) as optional ablation on best SFT adapter — only adopt if accuracy improves on validation split
- [ ] T066 [P] Create public documentation notebook at notebooks/submission_demo.ipynb (document methods, solvers, trace design, training setup — required for prize eligibility)
- [ ] T067 Update error_analysis.ipynb with final model comparison (baseline vs easy-SFT vs full-SFT vs best)
- [ ] T068 Verify reproducibility: re-run best experiment from committed config with same seed, confirm identical results
- [ ] T069 Select 2 final submissions for Kaggle (best overall + best on weakest category)
- [ ] T070 Final commit: all configs, results, traces, and documentation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Setup — BLOCKS all user stories
- **US1 Baseline (Phase 3)**: Depends on Foundational
- **US2 Easy Solvers (Phase 4)**: Can start after Foundational (independent of US1, but US1 provides comparison baseline)
- **US3 SFT & Hard Solvers (Phase 5)**: Depends on US2 (needs easy traces for first SFT); hard solvers (T042-T050) can start in parallel with easy SFT (T037-T041)
- **US4 Submission (Phase 6)**: Depends on US3 (needs best adapter)
- **Polish (Phase 7)**: Depends on US4

### Parallel Opportunities

**Phase 1 (Setup)**:
```text
Parallel: T003, T004, T005 (independent utility modules)
```

**Phase 4 (Easy Solvers)**:
```text
Parallel: T022+T023 (numeral), T025+T026 (gravity), T028+T029 (unit), T031+T032 (cipher)
— All 4 solver+trace pairs are independent of each other
```

**Phase 5 (Hard Solvers — while easy SFT trains)**:
```text
Parallel: T042+T043 (bit_manip), T045+T046 (equation), T048+T049 (cryptarithm)
— All 3 hard solver pairs can run alongside T039 (easy SFT training)
```

**Phase 7 (Polish)**:
```text
Parallel: T065 (RL ablation), T066 (documentation notebook)
```

---

## Implementation Strategy

### MVP First (US1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: US1 Baseline Evaluation
4. **STOP and VALIDATE**: Baseline accuracy confirmed (~0.49)

### Fast Path to First Score

1. Setup + Foundational + US1 Baseline → ~0.49
2. US2 Easy Solvers → 4 verified solvers at 100%
3. US3 First SFT (easy only) → first submission ~0.67
4. **Submit to Kaggle** — validate pipeline works end-to-end

### Full Path to Competitive Score

1. US3 Hard Solvers → bit_manip (85%), equation (80%), cryptarithm (8%)
2. US3 Full SFT → ~0.85 on validation
3. US4 Ablation & Optimization → squeeze final gains
4. Submit best adapter → target 0.85+ on leaderboard

### Critical Path

```text
T001-T007 → T008-T015 → T016-T021 → T022-T036 → T037-T055 → T062-T064
(Setup)     (Foundation) (Baseline)  (Solvers)    (SFT+Hard)  (Submit)
```

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story
- Solvers are CPU-only Python — no GPU needed until training (T039)
- Each solver MUST be verified 100% (easy) or to expected rate (hard) on train.csv before generating traces
- All traces MUST fit within 7680 token limit
- All evaluation MUST use competition params (temp=0, enable_thinking=True)
- Commit config + results after each experiment per constitution
- LoRA rank MUST be <= 32, target_modules: `in_proj|out_proj|up_proj|down_proj`
- Winner's public traces (Nemotron-cot-Tong dataset) available as reference/fallback
