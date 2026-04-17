# Deep Research: Competition Intelligence

**Collected**: 2026-04-16 via Playwright from Kaggle & HuggingFace

## 1. Puzzle Categories (from "Answers To Everything" + Progress Prize)

The competition has **7 categories** in train.csv (9500 samples):

| Category | Count | Best Solve Rate | Approach |
|----------|-------|----------------|----------|
| bit_manipulation | 1602 | 85.1% | Per-bit boolean function identification, bit-serial gate computation |
| cipher | 1576 | 100% | Substitution cipher cracking, char-by-char decryption |
| cryptarithm_deduce | 659 | 8.2% | Verbal arithmetic, hardest category |
| cryptarithm_guess | 164 | 6.7% | Verbal arithmetic variant, hardest |
| equation_numeric_deduce | 596 | 90.6% | Discover arithmetic operator from examples |
| equation_numeric_guess | 136 | 15.4% | Equation operator, guess variant |
| gravity | 1597 | 100% | Physics: d=0.5gt^2, derive g from examples |
| numeral | 1576 | 100% | Roman numeral conversion (1-100) |
| unit_conversion | 1594 | 100% | Linear scaling: derive factor from examples |

**Winner's target**: 87.7% (8333/9500). Achieved 85% on LB.
**Current top LB**: 0.86.

## 2. Winning Approach (Open Progress Prize, 0.877 target)

**Author**: Tong Hui Kang (121st). Key insights:

### Core Philosophy
- **SFT only, no RL needed**. The optimal policy is already known
  (deterministic chain-of-thought). The model just needs to follow it.
- **No distillation** from larger models. Optimal policy is generated
  with code.
- **Temperature = 0.0**: Only need the most likely token to be correct.
- **Objective**: Maximize minimum logprob across all training traces.

### Chain-of-Thought Design Principles
1. **Deterministic**: Each token derivable from as few sources as
   possible
2. **Simple**: Division broken into subtraction/addition steps
3. **Coverage**: Rare operations well-trained (don't stop at first
   match, iterate through all options)
4. **Within 7680 token limit**
5. **Tokenization-aware**: Avoid patterns where tokenizer merges
   letters/symbols
6. **Generalizable**: Not memorizing answers

### Training Details
- 27.8M tokens for winning solution, 599M total across experiments
- Trained on Tinker (NVIDIA platform)
- Cost: ~$212 Tinker + ~$60 Modal + $10 Kaggle/Colab
- Trainable params: 880M / 32.5B (2.71%)
- Required adapter conversion from Tinker format to submission format
  (SVD lossy conversion for Mamba in_proj layers)

### Per-Category Approaches
- **Numeral**: Enumerate all 1-100, step-by-step conversion
- **Gravity/Unit Conversion**: Rate-first decomposition, multi-step
  arithmetic, rate consistency verification
- **Cipher**: Build character mapping, handle unmapped chars via
  vocabulary matching (~90 Wonderland words)
- **Bit Manipulation**: Per-bit boolean function search through
  hierarchical gate levels (constants → identity → NOT → 2-input →
  3-input → 4-input gates)
- **Equation**: 4 operand transforms × 32 operators, frequency-ordered
  brute force scan
- **Cryptarithm**: Hardest, only ~7-8% solve rate. Concatenation/
  reverse concatenation detection.

## 3. Bit Manipulation Deep Dive

### Gate Hierarchy (52 total functions)
- Level 0 (2): Constants (all-0, all-1)
- Level 1 (2): Identity, NOT
- Level 2 (10): AND, OR, XOR, NAND, NOR, XNOR + 4 negation variants
- Level 3 (18): MAJ, CHO, PAR3, AO, OA, AX, OX, XA, XO + variants
- Level 4 (20): AOA, OAO, PAR4, XX, AXA + variants

### Distribution in train.csv
- identity: 39.9%
- CONST: 14.1%
- AND: 12.4%
- OR: 10.3%
- XOR: 7.9%
- XNOR: 6.8%
- NOT: 5.3%
- Rare 3/4-input gates: < 2% combined

### Critical Insight
The model MUST do bit-serial computation (spell out each operation
one bit at a time like `0&1=0 1&1=1`). Parallel multi-bit operations
cause accuracy to crater to 9.3%.

## 4. Metric Update (Critical)

A metric bug was fixed that caused binary answers to match as floats
instead of exact strings. This led to a ~0.3-0.4 point drop on new
submissions. All pre-March-28 submissions were rescored. The current
metric code (which we captured) is the authoritative version.

## 5. Key Discussion Findings

### "Do not distill from models that prohibit it"
- Competition host CPMP confirmed: using outputs from models that
  prohibit distillation (e.g., Gemini, GPT-5) violates competition
  rules
- Safe to use: open-source model outputs, self-generated data

### "Why GRPO is Painfully Slow on Nemotron (and the Fix)"
- GRPO is very slow on this architecture due to Mamba-2 layers
- Fix exists but RL may not be necessary (winner used SFT only)

### "Missing pieces: equation and cryptarithm"
- Cryptarithm is the hardest category with no known high-accuracy
  approach
- Some equation types have undiscovered operators

### Score Variance (0.84-0.86)
- Same adapter can produce different scores across submissions
- Likely due to the 50/50 public/private split and randomness in
  which problems are in each half

## 6. Tokenizer & Chat Template

### Format: ChatML-style
```
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
<think>
{reasoning}
</think>
{final_answer}<|im_end|>
```

### Special Tokens
| ID | Token | Type |
|----|-------|------|
| 0 | `<unk>` | special |
| 1 | `<s>` | special |
| 2 | `</s>` | special |
| 10 | `<\|im_start\|>` | special |
| 11 | `<\|im_end\|>` | special |
| 12 | `<think>` | **regular** (not special) |
| 13 | `</think>` | **regular** (not special) |

### Thinking Mode
- `enable_thinking=True` (default): Model produces `<think>...</think>`
  before the answer
- Competition uses `enable_thinking=True`
- `<think>` and `</think>` are regular tokens, meaning they can be
  part of training data naturally

## 7. Rules Highlights

- Max team size: 5
- Max 5 submissions/day, 2 final submissions
- External data allowed if publicly available and equally accessible
- Winner must publish under CC BY 4.0
- Winner must provide reproducible methodology description
- Cannot use outputs from models prohibiting distillation

## 8. Revised Strategy Implications

Based on this deep research, our plan needs fundamental revision:

### What the Winner Did (and we should learn from)
1. **Programmatically generated chain-of-thought traces** for each
   puzzle category — NOT using existing datasets
2. **SFT only** — no RL, no distillation, no OpenMathReasoning
3. **Category-specific reasoning templates** — each puzzle type has
   its own step-by-step format
4. **Bit-serial computation** for binary — spelling out each gate
   operation one bit at a time
5. **Maximize minimum logprob** as training objective
6. **Within 7680 token limit** — traces must fit

### What This Means for Our Plan
1. **OpenMathReasoning is largely irrelevant** — the puzzles are
   domain-specific, not standard math
2. **Prompt engineering matters less** than crafting the right CoT
   traces in code
3. **The primary work is writing Python code** that generates correct
   reasoning traces for each of the 7 puzzle categories
4. **SFT is sufficient** — RL is optional and may not help
5. **The bottleneck is understanding each puzzle type** and writing
   a solver + trace generator for it
6. **100% solve rate is achievable** on 4 categories (cipher, gravity,
   numeral, unit_conversion) = 6343/9500 = 66.8% floor
7. **Bit manipulation** (85% achievable) adds ~1362 = 80.1% total
8. **Equations** (76-90%) add ~500-540 = 85-86% total
9. **Cryptarithm** (~8%) adds ~65 = 86-87% total
