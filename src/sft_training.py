"""Supervised Fine-Tuning (SFT) — two stages.

Stage 1 (sft_base): Train base model on UltraChat for general chat ability.
  - Trained ONCE, shared by all DPO conditions.
  - Already completed and saved to models/sft_base/ on Modal.

Stage 2 (sft_ideology): Train SFT model on PoliTune ideological data.
  - Per-condition: right-leaning, left-leaning, or mixed.
  - Teaches the model to generate in the target ideological style.
  - Merges LoRA into base and saves full model for optional DPO on top.

Both stages use QLoRA with the Zephyr recipe hyperparameters.
"""

from __future__ import annotations

import gc
from pathlib import Path

import torch
from datasets import DatasetDict
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def _run_sft(
    train_data,
    eval_data,
    base_model_path: str,
    output_dir: str,
    run_name: str,
    config: dict,
) -> str:
    """Shared SFT training logic for both stages.

    Args:
        train_data: HF Dataset for training.
        eval_data: HF Dataset for evaluation.
        base_model_path: Path or HF model ID to load as base.
        output_dir: Where to save the merged model.
        run_name: Name for W&B run.
        config: Training configuration dict.

    Returns:
        Path to saved merged model.
    """
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
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
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
        wandb.init(
            project=train_cfg["wandb_project"],
            name=run_name,
            config={"stage": "sft", "base_model": base_model_path, "run_name": run_name},
        )

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=sft_cfg.get("num_epochs", 1),
        per_device_train_batch_size=sft_cfg.get("per_device_batch_size", 4),
        per_device_eval_batch_size=sft_cfg.get("per_device_batch_size", 4),
        gradient_accumulation_steps=sft_cfg.get("gradient_accumulation_steps", 2),
        learning_rate=sft_cfg.get("learning_rate", 2e-4),
        warmup_steps=int(((len(train_data) // (sft_cfg.get("per_device_batch_size", 4) * sft_cfg.get("gradient_accumulation_steps", 2))) * sft_cfg.get("num_epochs", 1)) * sft_cfg.get("warmup_ratio", 0.1)),
        max_length=sft_cfg.get("max_length", 2048),
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        seed=train_cfg["seed"],
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

    output_path = Path(output_dir)
    checkpoints = sorted(output_path.glob("checkpoint-*")) if output_path.exists() else []
    if checkpoints:
        print(f"Resuming from checkpoint: {checkpoints[-1]}")
        trainer.train(resume_from_checkpoint=str(checkpoints[-1]))
    else:
        trainer.train()

    print("Merging LoRA adapter into base model...")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if wandb_enabled:
        import wandb
        wandb.finish()

    del trainer, model, merged_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Saved merged SFT model to {output_dir}")
    return output_dir


def train_sft_base(config: dict) -> str:
    """Stage 1: Train base model on UltraChat for general chat ability."""
    from datasets import load_dataset

    train_cfg = config["training"]
    sft_cfg = train_cfg.get("sft", {})
    output_dir = str(Path(config["paths"]["models_dir"]) / "sft_base")

    if Path(f"{output_dir}/config.json").exists():
        print(f"SFT base model already exists at {output_dir}, skipping")
        return output_dir

    subset_size = sft_cfg.get("subset_size", 20000)
    print(f"Loading UltraChat 200K (subsample {subset_size})...")
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    dataset = dataset.shuffle(seed=train_cfg["seed"]).select(range(subset_size))
    dataset = dataset.remove_columns([c for c in dataset.column_names if c != "messages"])

    split = dataset.train_test_split(test_size=0.05, seed=train_cfg["seed"])

    return _run_sft(
        split["train"], split["test"],
        base_model_path=train_cfg["base_model"],
        output_dir=output_dir,
        run_name="sft_base",
        config=config,
    )


def train_sft_ideology(sft_dataset: DatasetDict, config: dict, condition_name: str) -> str:
    """Stage 2: Train SFT model on ideological data (PoliTune).

    Loads the Stage 1 SFT model and continues training on politically
    biased data to shift the model's ideological position.

    Args:
        sft_dataset: DatasetDict with train/eval splits (messages format).
        config: Configuration dict.
        condition_name: e.g. "sft_right", "sft_left", "sft_merged".

    Returns:
        Path to saved merged model.
    """
    sft_base_model = config["training"].get("sft_base_model", "mistralai/Mistral-7B-Instruct-v0.2")
    output_dir = str(Path(config["paths"]["models_dir"]) / condition_name)

    if Path(f"{output_dir}/config.json").exists():
        print(f"{condition_name} model already exists at {output_dir}, skipping")
        return output_dir

    return _run_sft(
        sft_dataset["train"], sft_dataset["eval"],
        base_model_path=sft_base_model,
        output_dir=output_dir,
        run_name=condition_name,
        config=config,
    )
