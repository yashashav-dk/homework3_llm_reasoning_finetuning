"""Inference and evaluation pipeline using vLLM and the competition metric."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompts(puzzles: list[dict], tokenizer) -> list[str]:
    """Format each puzzle into a chat-templated prompt string.

    Appends the standard boxed-answer instruction, then applies the tokenizer's
    chat template with generation prompt and thinking enabled.
    """
    prompts: list[str] = []
    suffix = (
        "\nPlease put your final answer inside `\\boxed{}`. "
        "For example: `\\boxed{your answer}`"
    )
    for puzzle in puzzles:
        user_content = puzzle["prompt"] + suffix
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompts.append(formatted)
    return prompts


# ---------------------------------------------------------------------------
# vLLM inference
# ---------------------------------------------------------------------------

def run_inference(
    model_name: str,
    prompts: list[str],
    adapter_path: str | None = None,
    **kwargs: Any,
) -> list[str]:
    """Run batch inference with vLLM and return a list of generated text strings.

    Parameters
    ----------
    model_name:
        HuggingFace model id or local path.
    prompts:
        Pre-formatted prompt strings (already has chat template applied).
    adapter_path:
        Optional path to a LoRA adapter directory.  When provided the adapter
        is loaded via ``vllm.lora.request.LoRARequest``.
    **kwargs:
        Override default sampling parameters (temperature, top_p, max_tokens).
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    temperature = kwargs.get("temperature", 0.0)
    top_p = kwargs.get("top_p", 1.0)
    max_tokens = kwargs.get("max_tokens", 7680)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    llm_kwargs: dict[str, Any] = {
        "model": model_name,
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.85,
    }
    if adapter_path is not None:
        llm_kwargs["enable_lora"] = True

    llm = LLM(**llm_kwargs)

    lora_request: LoRARequest | None = None
    if adapter_path is not None:
        lora_request = LoRARequest(
            lora_name="adapter",
            lora_int_id=1,
            lora_local_path=adapter_path,
        )

    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )

    return [output.outputs[0].text for output in outputs]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(puzzles: list[dict], responses: list[str]) -> dict:
    """Score model responses against ground-truth answers.

    Parameters
    ----------
    puzzles:
        Original puzzle dicts.  Each must have ``"answer"``, ``"category"``,
        and ideally ``"id"`` / ``"puzzle_id"`` fields.
    responses:
        Raw generated text strings, one per puzzle.

    Returns
    -------
    dict with keys:
        ``overall_accuracy``, ``per_category_accuracy``,
        ``per_category_counts``, ``errors``.
    """
    from src.metrics.competition import extract_final_answer, verify

    category_counts: dict[str, dict[str, int]] = {}
    errors: list[dict] = []

    for puzzle, response in zip(puzzles, responses):
        category = puzzle.get("category", "unknown")
        puzzle_id = puzzle.get("id") or puzzle.get("puzzle_id", "unknown")
        expected = str(puzzle.get("answer", ""))

        predicted = extract_final_answer(response)
        correct = verify(expected, predicted)

        if category not in category_counts:
            category_counts[category] = {"correct": 0, "total": 0}
        category_counts[category]["total"] += 1
        if correct:
            category_counts[category]["correct"] += 1
        else:
            errors.append(
                {
                    "puzzle_id": puzzle_id,
                    "category": category,
                    "expected": expected,
                    "predicted": predicted,
                    "response_snippet": response[:200],
                }
            )

    total_correct = sum(v["correct"] for v in category_counts.values())
    total_puzzles = sum(v["total"] for v in category_counts.values())
    overall_accuracy = total_correct / total_puzzles if total_puzzles > 0 else 0.0

    per_category_accuracy = {
        cat: (counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0)
        for cat, counts in category_counts.items()
    }

    return {
        "overall_accuracy": overall_accuracy,
        "per_category_accuracy": per_category_accuracy,
        "per_category_counts": category_counts,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run vLLM inference and evaluate with the competition metric."
    )
    parser.add_argument(
        "--model",
        default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        help="Model name or path (default: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional path to a LoRA adapter directory.",
    )
    parser.add_argument(
        "--puzzles",
        default="data/puzzles_classified.jsonl",
        help="Path to puzzles JSONL file (default: data/puzzles_classified.jsonl)",
    )
    parser.add_argument(
        "--split",
        default="data/splits/val_ids.json",
        help="Path to val IDs JSON file (default: data/splits/val_ids.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write results JSON.  Printed to stdout when omitted.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config file to override inference parameters.",
    )
    args = parser.parse_args()

    import random
    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # Load optional YAML config overrides
    # ------------------------------------------------------------------
    inference_kwargs: dict[str, Any] = {}
    if args.config is not None:
        import yaml  # type: ignore[import]

        with open(args.config) as fh:
            config = yaml.safe_load(fh) or {}
        inference_kwargs.update(config.get("eval", config.get("inference", {})))

    # ------------------------------------------------------------------
    # Load puzzles
    # ------------------------------------------------------------------
    puzzles_path = Path(args.puzzles)
    if not puzzles_path.exists():
        print(f"ERROR: puzzles file not found: {puzzles_path}", file=sys.stderr)
        sys.exit(1)

    puzzles: list[dict] = []
    with open(puzzles_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                puzzles.append(json.loads(line))

    # ------------------------------------------------------------------
    # Filter to validation split
    # ------------------------------------------------------------------
    split_path = Path(args.split)
    if split_path.exists():
        with open(split_path) as fh:
            val_ids: set[str] = set(json.load(fh))

        id_key = "id" if puzzles and "id" in puzzles[0] else "puzzle_id"
        puzzles = [p for p in puzzles if str(p.get(id_key, "")) in val_ids]
        print(f"Filtered to {len(puzzles)} validation puzzles.", file=sys.stderr)
    else:
        print(
            f"WARNING: split file not found ({split_path}), using all puzzles.",
            file=sys.stderr,
        )

    if not puzzles:
        print("ERROR: no puzzles to evaluate.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Build prompts (lazy tokenizer import)
    # ------------------------------------------------------------------
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = build_prompts(puzzles, tokenizer)

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    print(
        f"Running inference on {len(prompts)} prompts with model {args.model} ...",
        file=sys.stderr,
    )
    responses = run_inference(
        model_name=args.model,
        prompts=prompts,
        adapter_path=args.adapter,
        **inference_kwargs,
    )

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    results = evaluate(puzzles, responses)

    # ------------------------------------------------------------------
    # Print per-category accuracy table
    # ------------------------------------------------------------------
    print("\n=== Per-Category Accuracy ===")
    header = f"{'Category':<30} {'Correct':>8} {'Total':>8} {'Accuracy':>10}"
    print(header)
    print("-" * len(header))
    for cat, acc in sorted(results["per_category_accuracy"].items()):
        counts = results["per_category_counts"][cat]
        print(
            f"{cat:<30} {counts['correct']:>8} {counts['total']:>8} {acc:>9.1%}"
        )
    print("-" * len(header))
    print(
        f"{'OVERALL':<30} "
        f"{sum(v['correct'] for v in results['per_category_counts'].values()):>8} "
        f"{sum(v['total'] for v in results['per_category_counts'].values()):>8} "
        f"{results['overall_accuracy']:>9.1%}"
    )
    print()

    # ------------------------------------------------------------------
    # Save / print results
    # ------------------------------------------------------------------
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"Results saved to {output_path}", file=sys.stderr)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
