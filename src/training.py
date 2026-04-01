"""DPO training with QLoRA for each experimental condition.

Step 5 of the pipeline. Trains DPO on Qwen 3.5 9B using:
  - 4-bit quantization (BitsAndBytes)
  - LoRA rank 16, alpha 32
  - DPO beta=0.1, lr=5e-6, 3 epochs

For the DPO-Multi condition, trains two separate LoRA adapters
(one on optimist data, one on skeptic data) that get routed at inference.

Each condition trains from a fresh base model to ensure fair comparison.
"""

from __future__ import annotations

import gc
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer


def setup_model_and_tokenizer(config: dict, model_path: str | None = None) -> tuple:
    """Load the model with QLoRA quantization.

    If model_path is provided (SFT checkpoint), loads from that path.
    Otherwise loads the base model from config.

    Args:
        config: Configuration dict (needs training section).
        model_path: Optional path to a merged SFT model to use instead of base.

    Returns:
        Tuple of (model, tokenizer) ready for DPO training.
    """
    train_cfg = config["training"]
    load_from = model_path or train_cfg["base_model"]

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model (either base or SFT checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        load_from,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    # Prepare for LoRA training
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        load_from,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _get_lora_config(config: dict) -> LoraConfig:
    """Create LoRA config from training parameters."""
    train_cfg = config["training"]
    return LoraConfig(
        r=train_cfg["lora_rank"],
        lora_alpha=train_cfg["lora_alpha"],
        lora_dropout=train_cfg["lora_dropout"],
        target_modules=train_cfg["lora_target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def train_dpo(
    dataset: DatasetDict,
    config: dict,
    condition_name: str,
) -> str:
    """Train a single DPO condition and save the adapter.

    Uses TRL's DPOTrainer with the hyperparameters from config.

    Args:
        dataset: HuggingFace DatasetDict with train/eval splits.
        config: Configuration dict.
        condition_name: Name of the condition (for saving the adapter).

    Returns:
        Path to the saved LoRA adapter directory.
    """
    train_cfg = config["training"]
    output_dir = str(Path(config["paths"]["models_dir"]) / condition_name)

    # Check for unified SFT checkpoint to use as base
    sft_path = Path(config["paths"]["models_dir"]) / "sft_base"
    if sft_path.exists() and (sft_path / "config.json").exists():
        model_path = str(sft_path)
        print(f"Found SFT checkpoint: {model_path}")
    else:
        model_path = None

    print(f"\n{'='*60}")
    print(f"DPO Training: {condition_name}")
    print(f"  Base: {model_path or train_cfg['base_model']}")
    print(f"  Train: {len(dataset['train'])} pairs")
    print(f"  Eval:  {len(dataset['eval'])} pairs")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    # Load model (from SFT checkpoint if available, otherwise base)
    model, tokenizer = setup_model_and_tokenizer(config, model_path=model_path)
    lora_config = _get_lora_config(config)

    # DPO training config
    wandb_enabled = train_cfg.get("wandb_project") is not None
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        beta=train_cfg["dpo_beta"],
        max_length=train_cfg["max_length"],
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        eval_strategy="epoch",
        seed=train_cfg["seed"],
        report_to="wandb" if wandb_enabled else "none",
        run_name=condition_name if wandb_enabled else None,
        remove_unused_columns=False,
    )

    # Set W&B project name if enabled
    if wandb_enabled:
        import wandb
        wandb.init(
            project=train_cfg["wandb_project"],
            name=condition_name,
            config={
                "condition": condition_name,
                "base_model": train_cfg["base_model"],
                "lora_rank": train_cfg["lora_rank"],
                "dpo_beta": train_cfg["dpo_beta"],
                "learning_rate": train_cfg["learning_rate"],
                "num_epochs": train_cfg["num_epochs"],
                "train_pairs": len(dataset["train"]),
                "eval_pairs": len(dataset["eval"]),
            },
        )

    # Build trainer — ref_model=None means TRL creates a frozen copy automatically
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Check for existing checkpoints to resume from
    output_path = Path(output_dir)
    checkpoints = sorted(output_path.glob("checkpoint-*")) if output_path.exists() else []
    if checkpoints:
        resume_from = str(checkpoints[-1])
        print(f"Resuming from checkpoint: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    # Save the final LoRA adapter and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Close W&B run before cleanup
    if wandb_enabled:
        import wandb
        wandb.finish()

    # Clean up GPU memory
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Saved adapter to {output_dir}")
    return output_dir


def train_multi_adapter(
    multi_dataset: DatasetDict,
    config: dict,
) -> tuple[str, str]:
    """Train separate LoRA adapters for the DPO-Multi condition.

    Splits the multi dataset by the 'distribution' column and trains
    one adapter per distribution.

    Args:
        multi_dataset: DatasetDict with 'distribution' column.
        config: Configuration dict.

    Returns:
        Tuple of (optimist_adapter_path, skeptic_adapter_path).
    """
    train_data = multi_dataset["train"]
    eval_data = multi_dataset["eval"]

    # Split by distribution
    opt_train = train_data.filter(lambda x: x["distribution"] == "optimist")
    opt_eval = eval_data.filter(lambda x: x["distribution"] == "optimist")
    skp_train = train_data.filter(lambda x: x["distribution"] == "skeptic")
    skp_eval = eval_data.filter(lambda x: x["distribution"] == "skeptic")

    print(f"DPO-Multi split: optimist={len(opt_train)} train, skeptic={len(skp_train)} train")

    opt_ds = DatasetDict({"train": opt_train, "eval": opt_eval})
    skp_ds = DatasetDict({"train": skp_train, "eval": skp_eval})

    opt_path = train_dpo(opt_ds, config, "dpo_multi_optimist")
    skp_path = train_dpo(skp_ds, config, "dpo_multi_skeptic")

    return opt_path, skp_path


def train_all_conditions(
    datasets: dict[str, DatasetDict],
    config: dict,
) -> dict[str, str]:
    """Train DPO for all conditions sequentially.

    Handles dpo_multi specially by splitting into two sub-adapters.
    Cleans up GPU memory between runs.

    Args:
        datasets: Dict mapping condition name -> DatasetDict.
        config: Configuration dict.

    Returns:
        Dict mapping condition name -> adapter path.
    """
    adapter_paths = {}
    models_dir = Path(config["paths"]["models_dir"])

    for name, dataset in datasets.items():
        if name == "dpo_multi":
            # Check if both multi adapters are already complete
            opt_done = (models_dir / "dpo_multi_optimist" / "adapter_config.json").exists()
            skp_done = (models_dir / "dpo_multi_skeptic" / "adapter_config.json").exists()
            if opt_done and skp_done:
                print(f"Skipping dpo_multi — both adapters already complete")
                adapter_paths["dpo_multi_optimist"] = str(models_dir / "dpo_multi_optimist")
                adapter_paths["dpo_multi_skeptic"] = str(models_dir / "dpo_multi_skeptic")
            else:
                opt_path, skp_path = train_multi_adapter(dataset, config)
                adapter_paths["dpo_multi_optimist"] = opt_path
                adapter_paths["dpo_multi_skeptic"] = skp_path
        else:
            # Skip if adapter already fully trained (has adapter_config.json)
            adapter_file = models_dir / name / "adapter_config.json"
            if adapter_file.exists():
                print(f"Skipping {name} — adapter already complete")
                adapter_paths[name] = str(models_dir / name)
            else:
                adapter_paths[name] = train_dpo(dataset, config, name)

    return adapter_paths
