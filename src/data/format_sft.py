"""Format CoT traces as ChatML training examples for SFT."""

import argparse
import glob
import json
import pathlib
import statistics


def format_trace_to_sft(trace: dict, puzzle: dict) -> dict:
    """Format a single CoT trace and puzzle into a ChatML SFT example.

    Args:
        trace: dict with keys puzzle_id, category, thinking_text, final_answer, token_count
        puzzle: dict with keys id, prompt, answer

    Returns:
        dict with keys messages, puzzle_id, category, token_count
    """
    user_content = (
        f"{puzzle['prompt']}\n"
        "Please put your final answer inside `\\boxed{}`. "
        "For example: `\\boxed{your answer}`"
    )
    assistant_content = (
        f"<think>\n{trace['thinking_text']}\n</think>\n\n"
        f"\\boxed{{{trace['final_answer']}}}"
    )
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "puzzle_id": trace["puzzle_id"],
        "category": trace["category"],
        "token_count": trace["token_count"],
    }


def load_traces(trace_dir: str, categories: list[str] | None = None) -> list[dict]:
    """Load JSONL trace files from trace_dir.

    Args:
        trace_dir: directory containing files named {category}_traces.jsonl
        categories: if provided, only load traces for these categories

    Returns:
        list of all trace dicts
    """
    trace_path = pathlib.Path(trace_dir)
    traces: list[dict] = []

    if categories is not None:
        pattern_files = [trace_path / f"{cat}_traces.jsonl" for cat in categories]
    else:
        pattern_files = [
            pathlib.Path(p)
            for p in glob.glob(str(trace_path / "*_traces.jsonl"))
        ]

    for filepath in pattern_files:
        if not filepath.exists():
            continue
        with filepath.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    traces.append(json.loads(line))

    return traces


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format CoT traces as ChatML SFT training examples."
    )
    parser.add_argument(
        "--traces",
        default="data/traces/",
        help="Directory containing trace JSONL files (default: data/traces/)",
    )
    parser.add_argument(
        "--puzzles",
        default="data/puzzles_classified.jsonl",
        help="Puzzles JSONL file (default: data/puzzles_classified.jsonl)",
    )
    parser.add_argument(
        "--split",
        default="data/splits/train_ids.json",
        help="JSON file with train puzzle IDs (default: data/splits/train_ids.json)",
    )
    parser.add_argument(
        "--output",
        default="data/sft/train_sft.jsonl",
        help="Output JSONL file (default: data/sft/train_sft.jsonl)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=7680,
        help="Maximum token count filter (default: 7680)",
    )
    parser.add_argument(
        "--categories",
        default=None,
        help="Comma-separated list of categories to include (optional)",
    )
    args = parser.parse_args()

    # Parse categories
    categories: list[str] | None = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    # Load puzzles indexed by id
    puzzles: dict[str, dict] = {}
    puzzles_path = pathlib.Path(args.puzzles)
    with puzzles_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                puzzle = json.loads(line)
                puzzles[str(puzzle["id"])] = puzzle

    # Load train split IDs
    split_path = pathlib.Path(args.split)
    with split_path.open("r", encoding="utf-8") as f:
        train_ids: set[str] = {str(pid) for pid in json.load(f)}

    # Load traces
    traces = load_traces(args.traces, categories)

    # Filter to train split and max tokens, then format
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples: list[dict] = []
    skipped_split = 0
    skipped_tokens = 0
    skipped_missing = 0

    for trace in traces:
        puzzle_id = str(trace["puzzle_id"])

        if puzzle_id not in train_ids:
            skipped_split += 1
            continue

        if trace["token_count"] > args.max_tokens:
            skipped_tokens += 1
            continue

        if puzzle_id not in puzzles:
            skipped_missing += 1
            continue

        example = format_trace_to_sft(trace, puzzles[puzzle_id])
        examples.append(example)

    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    # Print summary
    print(f"Total examples written: {len(examples)}")
    print(f"  Skipped (not in train split): {skipped_split}")
    print(f"  Skipped (token count > {args.max_tokens}): {skipped_tokens}")
    print(f"  Skipped (puzzle not found): {skipped_missing}")

    # Per-category counts
    category_counts: dict[str, int] = {}
    for ex in examples:
        cat = ex["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    if category_counts:
        print("\nPer-category counts:")
        for cat, count in sorted(category_counts.items()):
            print(f"  {cat}: {count}")

    # Token stats
    if examples:
        token_counts = [ex["token_count"] for ex in examples]
        print("\nToken count stats:")
        print(f"  min:    {min(token_counts)}")
        print(f"  max:    {max(token_counts)}")
        print(f"  mean:   {statistics.mean(token_counts):.1f}")
        print(f"  median: {statistics.median(token_counts):.1f}")
        if len(token_counts) >= 2:
            print(f"  stdev:  {statistics.stdev(token_counts):.1f}")


if __name__ == "__main__":
    main()
