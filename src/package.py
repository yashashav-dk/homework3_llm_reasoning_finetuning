"""Package a LoRA adapter into a submission.zip for Kaggle."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

EXPECTED_TARGET_MODULES = {"in_proj", "out_proj", "up_proj", "down_proj"}
MAX_RANK = 32


def validate_adapter(adapter_dir: str) -> tuple[bool, list[str]]:
    """Validate a LoRA adapter directory.

    Checks:
    - adapter_config.json exists and is valid JSON
    - rank `r` is <= 32
    - target_modules references in_proj, out_proj, up_proj, down_proj
    - adapter_model.safetensors exists

    Returns:
        (is_valid, list_of_issues)
    """
    issues: list[str] = []
    adapter_path = Path(adapter_dir)

    # Check adapter_config.json
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        issues.append("adapter_config.json not found")
        # Cannot check further config fields; still check safetensors
    else:
        try:
            with config_path.open("r", encoding="utf-8") as f:
                config = json.load(f)
        except json.JSONDecodeError as exc:
            issues.append(f"adapter_config.json is not valid JSON: {exc}")
            config = None

        if config is not None:
            # Verify rank
            rank = config.get("r")
            if rank is None:
                issues.append("adapter_config.json missing field 'r' (rank)")
            elif not isinstance(rank, int):
                issues.append(f"adapter_config.json field 'r' must be an integer, got {type(rank).__name__}")
            elif rank > MAX_RANK:
                issues.append(f"rank r={rank} exceeds maximum allowed rank of {MAX_RANK}")

            # Verify target_modules
            target_modules = config.get("target_modules")
            if target_modules is None:
                issues.append("adapter_config.json missing field 'target_modules'")
            elif not isinstance(target_modules, list):
                issues.append(
                    f"adapter_config.json field 'target_modules' must be a list, got {type(target_modules).__name__}"
                )
            else:
                target_set = set(target_modules)
                missing = EXPECTED_TARGET_MODULES - target_set
                if missing:
                    issues.append(
                        f"target_modules is missing expected projection modules: {sorted(missing)}"
                    )

    # Check adapter_model.safetensors
    safetensors_path = adapter_path / "adapter_model.safetensors"
    if not safetensors_path.exists():
        issues.append("adapter_model.safetensors not found")

    is_valid = len(issues) == 0
    return is_valid, issues


def package_adapter(adapter_dir: str, output_path: str) -> str:
    """Package a LoRA adapter into a ZIP file for Kaggle submission.

    Validates the adapter first, then creates a ZIP containing:
    - adapter_config.json
    - adapter_model.safetensors

    Args:
        adapter_dir: Path to the adapter directory.
        output_path: Destination path for the output ZIP file.

    Returns:
        The output path as a string.

    Raises:
        ValueError: If the adapter fails validation.
    """
    is_valid, issues = validate_adapter(adapter_dir)
    if not is_valid:
        raise ValueError(
            "Adapter validation failed with the following issues:\n"
            + "\n".join(f"  - {issue}" for issue in issues)
        )

    adapter_path = Path(adapter_dir)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(adapter_path / "adapter_config.json", arcname="adapter_config.json")
        zf.write(adapter_path / "adapter_model.safetensors", arcname="adapter_model.safetensors")

    return str(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package a LoRA adapter into a submission.zip for Kaggle."
    )
    parser.add_argument(
        "--adapter",
        required=True,
        metavar="DIR",
        help="Path to the LoRA adapter directory.",
    )
    parser.add_argument(
        "--output",
        default="submissions/submission.zip",
        metavar="PATH",
        help="Output ZIP file path (default: submissions/submission.zip).",
    )
    args = parser.parse_args()

    # Validate first and surface any issues clearly
    is_valid, issues = validate_adapter(args.adapter)
    if not is_valid:
        print("Validation FAILED:")
        for issue in issues:
            print(f"  - {issue}")
        raise SystemExit(1)

    print("Validation passed.")

    output_path = package_adapter(args.adapter, args.output)

    # Print summary
    adapter_path = Path(args.adapter)
    config_path = adapter_path / "adapter_config.json"
    safetensors_path = adapter_path / "adapter_model.safetensors"
    out_path = Path(output_path)

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    rank = config.get("r")
    target_modules = config.get("target_modules", [])

    config_size_kb = config_path.stat().st_size / 1024
    weights_size_mb = safetensors_path.stat().st_size / (1024 * 1024)
    zip_size_mb = out_path.stat().st_size / (1024 * 1024)

    print(f"\nPackaged adapter summary:")
    print(f"  Rank (r):           {rank}")
    print(f"  Target modules:     {target_modules}")
    print(f"  adapter_config.json {config_size_kb:.1f} KB")
    print(f"  adapter_model.safetensors: {weights_size_mb:.2f} MB")
    print(f"  Output ZIP:         {output_path} ({zip_size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
