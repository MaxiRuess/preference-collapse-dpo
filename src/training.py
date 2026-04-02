"""DPO training with QLoRA for each experimental condition.

Trains DPO on the SFT base model (UltraChat-trained) using political
preference pairs from PoliTune. Each condition gets its own LoRA adapter.

Conditions:
  - dpo_right: right-leaning preferences
  - dpo_left: left-leaning preferences
  - dpo_merged: 50/50 contradictory labels (tests preference collapse)
"""

from __future__ import annotations

import gc
from pathlib import Path

import torch
from datasets import DatasetDict
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer


def setup_model_and_tokenizer(config: dict, model_path: str | None = None) -> tuple:
    """Load model with QLoRA quantization.

    Args:
        config: Configuration dict.
        model_path: Optional path to SFT checkpoint. Falls back to base model.

    Returns:
        Tuple of (model, tokenizer).
    """
    train_cfg = config["training"]
    load_from = model_path or train_cfg["base_model"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        load_from,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(load_from, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def train_dpo(
    dataset: DatasetDict,
    config: dict,
    condition_name: str,
) -> str:
    """Train a single DPO condition and save the adapter.

    Loads the SFT base model (if available) and trains a LoRA adapter
    using DPO preference pairs.

    Args:
        dataset: DatasetDict with train/eval splits.
        config: Configuration dict.
        condition_name: Name of the condition (for saving).

    Returns:
        Path to the saved LoRA adapter directory.
    """
    train_cfg = config["training"]
    output_dir = str(Path(config["paths"]["models_dir"]) / condition_name)

    # Use pre-built SFT model from HuggingFace Hub
    model_path = config["training"].get("sft_base_model")

    print(f"\n{'='*60}")
    print(f"DPO Training: {condition_name}")
    print(f"  Base: {model_path or train_cfg['base_model']}")
    print(f"  Train: {len(dataset['train'])} pairs")
    print(f"  Eval:  {len(dataset['eval'])} pairs")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    model, tokenizer = setup_model_and_tokenizer(config, model_path=model_path)

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
            name=condition_name,
            config={
                "stage": "dpo",
                "condition": condition_name,
                "dpo_beta": train_cfg["dpo_beta"],
                "learning_rate": train_cfg["learning_rate"],
                "train_pairs": len(dataset["train"]),
            },
        )

    # Calculate warmup_steps from ratio (warmup_ratio deprecated in TRL v5.2)
    total_steps = (len(dataset["train"]) // (train_cfg["per_device_batch_size"] * train_cfg["gradient_accumulation_steps"])) * train_cfg["num_epochs"]
    warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.1))

    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_steps=warmup_steps,
        beta=train_cfg["dpo_beta"],
        loss_type=train_cfg.get("loss_type", "ipo"),
        max_length=train_cfg["max_length"],
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        eval_strategy="epoch",
        seed=train_cfg["seed"],
        report_to="wandb" if wandb_enabled else "none",
        run_name=condition_name if wandb_enabled else None,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
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

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    if wandb_enabled:
        import wandb
        wandb.finish()

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Saved adapter to {output_dir}")
    return output_dir
