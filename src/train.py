"""QLoRA fine-tuning using PEFT and TRL SFTTrainer.

Heavy GPU-only imports (torch, transformers, peft, trl, bitsandbytes) are
deferred inside functions so this module can be safely imported on CPU-only
machines without errors.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_for_training(
    model_name: str,
    config: dict,
) -> tuple[Any, Any]:
    """Load a causal LM with 4-bit QLoRA quantisation and PEFT LoRA adapters.

    Args:
        model_name: HuggingFace model hub identifier or local path.
        config: Training config dict.  Recognised keys under ``lora``:
            r, lora_alpha, target_modules, lora_dropout.

    Returns:
        (model, tokenizer) tuple ready for SFTTrainer.
    """
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    lora_cfg = config.get("lora", {})

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    target_modules_raw = lora_cfg.get(
        "target_modules", r".*\.(in_proj|out_proj|up_proj|down_proj)$"
    )
    # Accept either a regex string or an explicit list of module names.
    if isinstance(target_modules_raw, str):
        # Convert regex to the list of matching module names expected by PEFT.
        all_module_names = [name for name, _ in model.named_modules()]
        target_modules: list[str] | str = [
            name.split(".")[-1]
            for name in all_module_names
            if re.fullmatch(target_modules_raw, name)
        ]
        if not target_modules:
            # Fallback: pass the raw string and let PEFT handle it.
            target_modules = target_modules_raw
    else:
        target_modules = list(target_modules_raw)

    lora_config = LoraConfig(
        r=int(lora_cfg.get("r", 32)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 16)),
        target_modules=target_modules,
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _load_jsonl_dataset(path: str | Path):
    """Load a JSONL file where each line is ``{"messages": [...]}``."""
    from datasets import Dataset

    records: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def train(config: dict) -> dict:
    """Fine-tune a model with QLoRA using TRL SFTTrainer.

    Args:
        config: Flat or nested config dict.  Recognised top-level keys:
            model_name, data_path, output_dir, seed, and per-trainer keys
            (learning_rate, per_device_train_batch_size, …).

    Returns:
        Dict of training metrics reported by SFTTrainer.
    """
    from trl import SFTConfig, SFTTrainer

    try:
        from trl import DataCollatorForCompletionOnlyLM
        _has_collator = True
    except ImportError:
        # TRL >= 1.0 removed DataCollatorForCompletionOnlyLM;
        # use SFTConfig(completions_only=True) instead
        _has_collator = False

    model_name: str = config["model_name"]
    data_path: str = config["data_path"]
    output_dir: str = config.get("output_dir", "outputs/adapter")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    logger.info("Loading training data from %s", data_path)
    dataset = _load_jsonl_dataset(data_path)
    logger.info("Loaded %d examples", len(dataset))

    # -- Validation gate: dataset --
    if len(dataset) == 0:
        raise ValueError("FATAL: dataset is empty — nothing to train on")
    sample = dataset[0]
    if "messages" not in sample:
        raise ValueError(
            f"FATAL: first record missing 'messages' key, got: {list(sample.keys())}"
        )
    msgs = sample["messages"]
    if len(msgs) < 2 or msgs[-1]["role"] != "assistant":
        raise ValueError(
            f"FATAL: expected messages ending with assistant turn, got roles: "
            f"{[m['role'] for m in msgs]}"
        )

    # ------------------------------------------------------------------
    # Model + tokenizer
    # ------------------------------------------------------------------
    logger.info("Loading model %s", model_name)
    model, tokenizer = load_model_for_training(model_name, config)

    # ------------------------------------------------------------------
    # SFTTrainer config
    # ------------------------------------------------------------------
    training_cfg = config.get("training", {})
    sft_config = SFTConfig(
        output_dir=output_dir,
        learning_rate=float(training_cfg.get("learning_rate", 2e-4)),
        per_device_train_batch_size=int(training_cfg.get("per_device_train_batch_size", 4)),
        gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 8)),
        max_seq_length=int(training_cfg.get("max_seq_length", 4096)),
        num_train_epochs=int(training_cfg.get("num_train_epochs", 1)),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.03)),
        bf16=True,
        gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", False)),
        logging_steps=int(training_cfg.get("logging_steps", 10)),
        save_strategy="epoch",
        dataset_text_field=None,
    )

    # Completion-only loss: only compute loss on assistant response tokens.
    response_template = "<|im_start|>assistant\n"

    collator = None
    if _has_collator:
        # TRL < 1.0: use explicit DataCollatorForCompletionOnlyLM
        response_token_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )
        if len(response_token_ids) == 0:
            raise ValueError(
                "FATAL: response_template tokenises to empty sequence — "
                "completion-only masking will fail"
            )
        logger.info(
            "Response template '%s' → %d token IDs: %s",
            response_template.replace("\n", "\\n"),
            len(response_token_ids),
            response_token_ids,
        )

        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=tokenizer,
        )

        # -- Validation gate: collator masks non-assistant tokens --
        _val_sample = tokenizer.apply_chat_template(
            dataset[0]["messages"], tokenize=False
        )
        _val_encoded = tokenizer(
            _val_sample,
            return_tensors="pt",
            truncation=True,
            max_length=sft_config.max_seq_length,
        )
        _val_batch = collator([{
            "input_ids": _val_encoded["input_ids"][0],
            "attention_mask": _val_encoded["attention_mask"][0],
        }])
        _val_labels = _val_batch["labels"][0]
        n_masked = (_val_labels == -100).sum().item()
        n_total = len(_val_labels)
        n_trained = n_total - n_masked
        if n_trained == 0:
            raise ValueError(
                "FATAL: collator masked ALL tokens — response template not found "
                "in tokenized example. Check chat template compatibility."
            )
        mask_pct = 100.0 * n_masked / n_total
        logger.info(
            "Collator validation: %d/%d tokens masked (%.1f%%), "
            "%d assistant tokens will receive loss",
            n_masked, n_total, mask_pct, n_trained,
        )
    else:
        # TRL >= 1.0: use SFTConfig completions_only flag
        sft_config.completions_only = True
        logger.info(
            "Using TRL >= 1.0 completions_only=True (no explicit collator)"
        )

    trainer_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": dataset,
        "args": sft_config,
    }
    if collator is not None:
        trainer_kwargs["data_collator"] = collator

    trainer = SFTTrainer(**trainer_kwargs)

    logger.info("Starting training …")
    train_result = trainer.train()
    metrics: dict = train_result.metrics

    # -- Validation gate: training metrics sanity --
    import math

    train_loss = metrics.get("train_loss", float("nan"))
    if math.isnan(train_loss) or math.isinf(train_loss):
        raise ValueError(
            f"FATAL: training loss is {train_loss} — model may have diverged"
        )
    if train_loss > 10.0:
        logger.warning(
            "Training loss %.4f is unusually high (>10) — "
            "model may not have learned effectively",
            train_loss,
        )
    logger.info("Training loss: %.4f (sanity check passed)", train_loss)

    # ------------------------------------------------------------------
    # Save adapter
    # ------------------------------------------------------------------
    logger.info("Saving LoRA adapter to %s", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # -- Validation gate: adapter files exist --
    output_path = Path(output_dir)
    expected_files = ["adapter_config.json", "adapter_model.safetensors"]
    missing = [f for f in expected_files if not (output_path / f).exists()]
    if missing:
        # Some versions use .bin instead of .safetensors
        if "adapter_model.safetensors" in missing and (
            output_path / "adapter_model.bin"
        ).exists():
            missing.remove("adapter_model.safetensors")
    if missing:
        raise ValueError(
            f"FATAL: adapter save failed — missing files in {output_dir}: {missing}"
        )
    logger.info(
        "Adapter validation passed: %s",
        [f.name for f in output_path.iterdir() if f.suffix in (".json", ".safetensors", ".bin")],
    )

    return metrics


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and run QLoRA fine-tuning."""
    from src.utils.config import config_hash, load_config
    from src.utils.logging import log_experiment
    from src.utils.seeds import set_seed

    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning with PEFT + TRL SFTTrainer"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML training config file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override (overrides value in config).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load and optionally patch config
    # ------------------------------------------------------------------
    config: dict = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed

    seed: int = int(config.get("seed", 42))
    set_seed(seed)

    cfg_hash = config_hash(config)
    run_id = str(uuid.uuid4())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("run_id=%s  config_hash=%s  seed=%d", run_id, cfg_hash, seed)
    logger.info("Config: %s", json.dumps(config, indent=2, default=str))

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    metrics = train(config)
    logger.info("Training complete. Metrics: %s", metrics)

    # ------------------------------------------------------------------
    # Log experiment
    # ------------------------------------------------------------------
    overall_accuracy = 0.0  # placeholder — real accuracy comes from evaluate.py
    log_experiment(
        run_id=run_id,
        config_hash=cfg_hash,
        hypothesis=config.get("hypothesis", ""),
        treatment_variable=config.get("treatment_variable", ""),
        seed=seed,
        categories_included=config.get("categories_included", []),
        overall_accuracy=overall_accuracy,
        per_category_accuracy=metrics.get("per_category_accuracy", {}),
        decision=config.get("decision", ""),
        control_run_id=config.get("control_run_id", ""),
    )
    logger.info("Experiment logged (run_id=%s)", run_id)


if __name__ == "__main__":
    main()
