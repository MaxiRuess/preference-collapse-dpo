"""Supervised fine-tuning (SFT) of Mistral-7B-Instruct-v0.2 on PoliTune data.

Local counterpart of modal_train.py. One run per (condition, seed):
  - QLoRA (rank 16, q_proj/v_proj, NF4) on the chosen responses
  - saves the LoRA adapter (generation adds its delta to the bf16 base)
  - writes train_summary.json with per-epoch losses for reporting
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from datasets import DatasetDict
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def latest_checkpoint(output_dir: str | Path) -> str | None:
    """Return the checkpoint dir with the highest step (numeric sort)."""
    p = Path(output_dir)
    if not p.exists():
        return None
    cps = [c for c in p.glob("checkpoint-*") if c.name.split("-")[-1].isdigit()]
    if not cps:
        return None
    return str(max(cps, key=lambda c: int(c.name.split("-")[-1])))


def _run_sft(
    train_data,
    eval_data,
    base_model_path: str,
    output_dir: str,
    adapter_dir: str,
    run_name: str,
    seed: int,
    config: dict,
) -> str:
    """Train one SFT run and save adapter + merged model."""
    train_cfg = config["training"]
    sft_cfg = train_cfg.get("sft", {})

    print(f"\n{'='*60}")
    print(f"SFT Training: {run_name}")
    print(f"  Base: {base_model_path}")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Eval:  {len(eval_data)} examples")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, quantization_config=bnb_config,
        device_map="auto", dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=train_cfg["lora_rank"],
        lora_alpha=train_cfg["lora_alpha"],
        lora_dropout=train_cfg["lora_dropout"],
        target_modules=train_cfg["lora_target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    wandb_enabled = train_cfg.get("wandb_project") is not None
    if wandb_enabled:
        import wandb
        wandb.init(project=train_cfg["wandb_project"], name=run_name,
                   config={"stage": "sft", "base_model": base_model_path,
                           "run_name": run_name, "seed": seed})

    bs = sft_cfg.get("per_device_batch_size", 4)
    ga = sft_cfg.get("gradient_accumulation_steps", 2)
    epochs = sft_cfg.get("num_epochs", 2)
    total_steps = (len(train_data) // (bs * ga)) * epochs
    warmup_steps = int(total_steps * sft_cfg.get("warmup_ratio", 0.1))

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        gradient_accumulation_steps=ga,
        learning_rate=sft_cfg.get("learning_rate", 2e-4),
        warmup_steps=warmup_steps,
        max_length=sft_cfg.get("max_length", 2048),
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        seed=seed,
        data_seed=seed,
        report_to="wandb" if wandb_enabled else "none",
        run_name=run_name if wandb_enabled else None,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    ckpt = latest_checkpoint(output_dir)
    if ckpt:
        print(f"Resuming from checkpoint: {ckpt}")
        trainer.train(resume_from_checkpoint=ckpt)
    else:
        trainer.train()

    history = trainer.state.log_history
    summary = {
        "run_name": run_name,
        "seed": seed,
        "n_train": len(train_data),
        "n_eval": len(eval_data),
        "total_steps": trainer.state.global_step,
        "warmup_steps": warmup_steps,
        "train_loss": [h for h in history if "loss" in h and "eval_loss" not in h],
        "eval_loss": [h for h in history if "eval_loss" in h],
        "hyperparams": sft_cfg,
    }

    # Save the adapter only; generation applies its dense delta to the bf16
    # base (src/generation.py), so no merged checkpoint is written.
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    Path(adapter_dir, "train_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Saved LoRA adapter + train_summary.json to {adapter_dir}")

    if wandb_enabled:
        import wandb
        wandb.finish()

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return adapter_dir


def train_sft_ideology(sft_dataset: DatasetDict, config: dict,
                       condition_name: str, seed: int) -> str:
    """Train one ideology SFT run (condition, seed) from the instruct base model.

    Args:
        sft_dataset: DatasetDict with train/eval splits (messages format).
        config: Configuration dict (configs/config.yaml).
        condition_name: "sft_right" | "sft_left" | "sft_merged".
        seed: Training seed; also selects the label-flip draw for sft_merged.

    Returns:
        Path to the saved adapter directory.
    """
    base = config["training"].get("sft_base_model", "mistralai/Mistral-7B-Instruct-v0.2")
    run_name = f"{condition_name}_s{seed}"
    models_dir = Path(config["paths"]["models_dir"])
    output_dir = str(models_dir / run_name)
    adapter_dir = str(models_dir / f"{run_name}_adapter")

    if Path(adapter_dir, "adapter_config.json").exists():
        print(f"{run_name} adapter already exists at {adapter_dir}, skipping")
        return adapter_dir

    return _run_sft(
        sft_dataset["train"].select_columns(["messages"]),
        sft_dataset["eval"].select_columns(["messages"]),
        base_model_path=base,
        output_dir=output_dir,
        adapter_dir=adapter_dir,
        run_name=run_name,
        seed=seed,
        config=config,
    )
