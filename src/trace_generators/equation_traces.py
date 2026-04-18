"""Chain-of-thought trace generator for equation puzzles."""

from __future__ import annotations

from src.solvers.base import SolverResult
from src.solvers.equation import _parse_examples, _find_rule, _rev
from src.trace_generators.base import BaseTraceGenerator, CoTTrace


class EquationTraceGenerator(BaseTraceGenerator):
    category = "equation"

    def generate_trace(self, puzzle: dict, solver_result: SolverResult) -> CoTTrace:
        puzzle_id = puzzle.get("id", "")
        prompt = puzzle.get("prompt", "")

        examples, target = _parse_examples(prompt)
        if not target or not examples:
            return CoTTrace(
                puzzle_id=puzzle_id, category=self.category,
                thinking_text="Unable to parse puzzle.", final_answer="",
            )

        target_a, target_sym, target_b = target

        # Group by symbol and find rule for target symbol
        sym_examples: dict[str, list[tuple[int, int, str]]] = {}
        for a, sym, b, result_str in examples:
            sym_examples.setdefault(sym, []).append((a, b, result_str))

        target_sym_examples = sym_examples.get(target_sym, [])
        rule = _find_rule(target_sym_examples) if target_sym_examples else None
        if rule is None:
            all_ex = [(a, b, r) for a, _, b, r in examples]
            rule = _find_rule(all_ex)

        lines = []
        lines.append("=== Step 1: Parse examples ===\n")
        for a, sym, b, r in examples:
            lines.append(f"  {a} {sym} {b} = {r}")
        lines.append(f"\n  Target: {target_a} {target_sym} {target_b} = ?\n")

        lines.append("=== Step 2: Find the rule ===\n")
        if rule is None:
            lines.append("  No matching rule found.\n")
            return CoTTrace(
                puzzle_id=puzzle_id, category=self.category,
                thinking_text="\n".join(lines), final_answer="",
            )

        op_func, ot_func, out_func, desc = rule
        lines.append(f"  Found rule: {desc}\n")

        lines.append("=== Step 3: Verify on examples ===\n")
        verify_examples = target_sym_examples if target_sym_examples else [(a, b, r) for a, _, b, r in examples]
        for a, b, result_str in verify_examples:
            try:
                predicted = str(out_func(op_func(ot_func(a), ot_func(b))))
            except Exception:
                predicted = "ERROR"
            match = "OK" if predicted.lstrip('0') == result_str.lstrip('0') else "MISMATCH"
            lines.append(f"  {a}, {b} -> {predicted} (expected {result_str}) {match}")
        lines.append("")

        lines.append("=== Step 4: Apply to target ===\n")
        try:
            raw = op_func(ot_func(target_a), ot_func(target_b))
            final = str(out_func(raw))
        except Exception:
            final = ""
        lines.append(f"  {target_a} {target_sym} {target_b} = {final}\n")

        return CoTTrace(
            puzzle_id=puzzle_id,
            category=self.category,
            thinking_text="\n".join(lines),
            final_answer=final,
            token_count=self.count_tokens("\n".join(lines)),
            is_verified=solver_result.is_correct,
        )


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/traces/")
    args = parser.parse_args()

    from src.solvers.equation import EquationSolver

    solver = EquationSolver()
    gen = EquationTraceGenerator()
    traces = []

    with open(args.input) as f:
        for line in f:
            p = json.loads(line)
            if not p["category"].startswith("equation"):
                continue
            result = solver.solve(p)
            if not result.is_correct:
                continue
            trace = gen.generate_trace(p, result)
            if trace.final_answer:
                traces.append({
                    "puzzle_id": trace.puzzle_id,
                    "category": trace.category,
                    "thinking_text": trace.thinking_text,
                    "final_answer": trace.final_answer,
                    "token_count": trace.token_count,
                    "is_verified": trace.is_verified,
                })

    import os
    os.makedirs(args.output, exist_ok=True)
    outfile = os.path.join(args.output, "equation_traces.jsonl")
    with open(outfile, "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")
    print(f"Wrote {len(traces)} equation traces to {outfile}")
