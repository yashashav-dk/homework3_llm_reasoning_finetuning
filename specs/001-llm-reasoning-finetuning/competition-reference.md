# Competition Reference: Collected from Kaggle & HuggingFace

**Collected**: 2026-04-16

## 1. Competition Metric (from NVIDIA Nemotron Metric notebook)

### Key Functions

#### `extract_final_answer(text)`
Extracts final answer from model response:
1. Priority: `\boxed{...}` — finds all matches, returns last non-empty
2. Fallback patterns: "The final answer is:", "Final answer is:", etc.
3. Last resort: last numeric value in text (`-?\d+(?:\.\d+)?`)
4. Ultimate fallback: last non-empty line of text
5. Returns `'NOT_FOUND'` if text is None

#### `verify(stored_answer, predicted)`
Compares predicted vs ground truth:
- Binary strings (`[01]+`): strict case-insensitive match (no leading-zero tolerance)
- Numeric: `math.isclose(rel_tol=1e-2, abs_tol=1e-5)`
- Otherwise: case-insensitive string comparison

**IMPORTANT**: The actual competition uses `rel_tol=1e-2` (not 1e-4 as
stated in the overview text). The code is authoritative.

#### `score()` default parameters (from metric notebook):
```python
max_lora_rank = 32
max_tokens = 3584      # NOTE: different from overview page (7680)
top_p = 1.0
temperature = 1.0      # NOTE: different from overview page (0.0)
max_num_seqs = 128      # NOTE: different from overview page (64)
gpu_memory_utilization = 0.85
max_model_len = 4096    # NOTE: different from overview page (8192)
```

**WARNING**: The `score()` function signature has different defaults than
the overview page states. The overview page parameters are passed as
explicit arguments when the metric is called, overriding these defaults.
The actual competition parameters (from the overview) are:
```
max_lora_rank = 32
max_tokens = 7680
top_p = 1.0
temperature = 0.0
max_num_seqs = 64
gpu_memory_utilization = 0.85
max_model_len = 8192
```

### Prompt Construction (from metric)
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

Key findings:
- `enable_thinking=True` is used — the model's reasoning/thinking mode
  is explicitly enabled during competition evaluation
- The prompt appends boxed format instructions to the raw puzzle prompt
- Uses `apply_chat_template` with a single user message
- Falls back to raw prompt if chat template fails

## 2. Submission Demo (from NVIDIA Nemotron Submission Demo notebook)

### Model Loading
```python
MODEL_PATH = kagglehub.model_download(
    "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,
)
```

### LoRA Configuration
```python
lora_config = LoraConfig(
    r=LORA_RANK,          # Can be set to a maximum of 32
    lora_alpha=16,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

**Critical**: Target modules use regex pattern matching:
`in_proj`, `out_proj`, `up_proj`, `down_proj`

### Saving & Packaging
```python
model.save_pretrained(OUTPUT_DIR)
subprocess.run("zip -m submission.zip *", shell=True, check=True)
```

Output files in zip: adapter_config.json, adapter_model.safetensors,
README.md, __notebook__.ipynb

### Submission Demo Score: 0.49 (public), 0.50 (best V11)
Runtime: 16m 43s on GPU RTX Pro 6000
Trainable params: 880,138,240 / 32,458,075,584 (2.71%)

## 3. Competition Dataset

- **train.csv**: 9500 rows, 3 columns (id, prompt, answer)
- **test.csv**: 3 sample rows (replaced by several hundred at scoring)
- **Size**: 3.07 MB total
- **License**: CC BY 4.0
- **Domain**: Logical reasoning puzzles — bit manipulation rules,
  algebraic equations, secret encryption rules
- **Format**: Each prompt describes a transformation rule with examples,
  asks to apply it to a new input

### Sample Problem Types (from test.csv preview)
- Bit manipulation: "In Alice's Wonderland, a secret bit manipulation
  rule transforms 8-bit binary numbers..."
- Encryption: "In Alice's Wonderland, secret encryption rules are
  used on text..."

## 4. Leaderboard (as of 2026-04-16)

- **#1**: 0.86 accuracy (Amantortopus)
- **#2-5**: 0.86 accuracy
- **#6-10**: 0.86-0.85 accuracy
- **#14**: 0.84 accuracy
- **2,057 teams**, 13,726 submissions
- Public leaderboard = 50% of test data; final = other 50%

## 5. Key Corrections to Our Plan

Based on competition metric source code:

1. **`enable_thinking=True`** — competition explicitly enables thinking
   mode. Our training data and prompts should account for this.
2. **Target modules** for LoRA: `in_proj`, `out_proj`, `up_proj`,
   `down_proj` (regex pattern, not the q/k/v/o_proj we assumed)
3. **Binary string handling** — the metric treats binary answers
   (`[01]+`) specially with strict comparison (no leading zero tolerance)
4. **Tolerance is 1e-2** in the actual code (not 1e-4 as overview states)
5. **Problem domain** is NOT math/code/logic reasoning from standard
   benchmarks — it's novel puzzle-solving (bit manipulation, encryption
   rules, algebraic equations). OpenMathReasoning may be less directly
   relevant than we assumed.

## 6. Model Card — Chat Template & Usage

### Transformers Usage (thinking ON — default)
```python
tokenizer = AutoTokenizer.from_pretrained(
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
)
model = AutoModelForCausalLM.from_pretrained(
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

messages = [{"role": "user", "content": "..."}]
tokenized_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)
```

### Thinking Mode Control
- `enable_thinking=True` (default): Model produces `<think>...</think>`
  reasoning before the final answer
- `enable_thinking=False`: Disables internal reasoning, direct answer
- Competition metric uses `enable_thinking=True`

### Thinking Format
The model wraps reasoning in `<think>...</think>` tags. The final
answer comes after `</think>`. Budget control can limit reasoning
tokens.

### vLLM Serving
```bash
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --served-model-name model \
  --max-num-seqs 8 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --port 8000 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser-plugin nano_v3_reasoning_parser.py \
  --reasoning-parser nano_v3
```

### LoRA Target Modules (from submission demo)
```python
target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$"
```

### Architecture Summary
- 52 layers: 23 Mamba-2 + 23 MoE + 6 Attention (GQA, 2 groups)
- MoE: 128 routed experts + 1 shared, 6 active per token
- 30B total params, 3.5B active per token
- Context: 256K default, up to 1M
- Languages: English, German, Spanish, French, Italian, Japanese
- Training: 25T tokens, WSD LR schedule, peak LR 1e-3

## 7. Critical Plan Revisions Needed

Based on all collected information, these changes are required:

1. **LoRA target modules**: Change from `[q_proj, k_proj, v_proj,
   o_proj]` to regex `r".*\.(in_proj|out_proj|up_proj|down_proj)$"`
   as shown in the official submission demo.

2. **enable_thinking=True**: Competition enables thinking mode.
   Training data should include `<think>...</think>` reasoning traces
   or the model should be fine-tuned to produce them.

3. **Problem domain**: The benchmark is novel logical puzzles (bit
   manipulation, encryption, algebraic equations), NOT standard math
   benchmarks. OpenMathReasoning CoT data may still help with general
   reasoning skills, but domain-specific training on the 9500 train
   samples is likely more impactful.

4. **Train on competition data**: The train.csv has 9500 puzzle-answer
   pairs. These should be the PRIMARY fine-tuning data, potentially
   augmented with OpenMathReasoning for general reasoning capacity.

5. **Tolerance is 1e-2**: The verify() function uses `rel_tol=1e-2`,
   not 1e-4 as the overview text states.

6. **Binary answer handling**: Binary strings are compared strictly
   (no leading-zero tolerance). The model must produce exact binary
   output for bit manipulation puzzles.

7. **Prompt format**: The competition appends
   `'\nPlease put your final answer inside \boxed{}...'` to the raw
   puzzle prompt and uses `apply_chat_template` with
   `enable_thinking=True`.
