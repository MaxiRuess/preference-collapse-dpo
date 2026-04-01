"""Supervised Fine-Tuning (SFT) on UltraChat — unified base for all conditions.

Step 5a of the pipeline. Trains Qwen 3.5 9B Base on the UltraChat 200K dataset
to produce a general-purpose chat model. This is trained ONCE and shared by all
DPO conditions as their starting point.

Follows the Zephyr pattern:
  - SFT teaches the base model to follow instructions and chat coherently
  - DPO (step 5b) then shifts the model's ideological preferences

After SFT, the LoRA adapter is merged into the base model to produce a full
checkpoint that DPO can load as its starting point.
"""

from __future__ import annotations

import gc
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def train_sft(config: dict) -> str:
    """Train unified SFT on UltraChat and save the merged model.

    Loads the base model with QLoRA, trains SFT on a subset of UltraChat,
    then merges the LoRA adapter back into the base model.

    Args:
        config: Configuration dict.

    Returns:
        Path to the saved merged model directory.
    """
    train_cfg = config["training"]
    sft_cfg = train_cfg.get("sft", {})
    output_dir = str(Path(config["paths"]["models_dir"]) / "sft_base")

    # Skip if already complete
    if Path(f"{output_dir}/config.json").exists():
        print(f"SFT model already exists at {output_dir}, skipping")
        return output_dir

    # Load UltraChat dataset
    sft_subset_size = sft_cfg.get("subset_size", 20000)
    print(f"Loading UltraChat 200K (subsample {sft_subset_size})...")
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    dataset = dataset.shuffle(seed=train_cfg["seed"]).select(range(sft_subset_size))
    # Remove prompt/prompt_id columns — keep only 'messages' so TRL
    # recognizes this as conversational language modeling format
    dataset = dataset.remove_columns([c for c in dataset.column_names if c != "messages"])

    # Split 95/5 for train/eval
    split = dataset.train_test_split(test_size=0.05, seed=train_cfg["seed"])
    train_data = split["train"]
    eval_data = split["test"]

    print(f"\n{'='*60}")
    print(f"SFT Training (unified)")
    print(f"  Dataset: UltraChat 200K ({sft_subset_size} subset)")
    print(f"  Train: {len(train_data)} conversations")
    print(f"  Eval:  {len(eval_data)} conversations")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    # Load base model with QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        train_cfg["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        train_cfg["base_model"],
        padding_side="right",
    )
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

    # W&B support
    wandb_enabled = train_cfg.get("wandb_project") is not None
    if wandb_enabled:
        import wandb
        wandb.init(
            project=train_cfg["wandb_project"],
            name="sft_base",
            config={
                "stage": "sft",
                "base_model": train_cfg["base_model"],
                "dataset": "HuggingFaceH4/ultrachat_200k",
                "subset_size": sft_subset_size,
                "learning_rate": sft_cfg.get("learning_rate", 2e-5),
                "num_epochs": sft_cfg.get("num_epochs", 1),
            },
        )

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=sft_cfg.get("num_epochs", 1),
        per_device_train_batch_size=sft_cfg.get("per_device_batch_size", 2),
        per_device_eval_batch_size=sft_cfg.get("per_device_batch_size", 2),
        gradient_accumulation_steps=sft_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=sft_cfg.get("learning_rate", 2e-5),
        warmup_ratio=sft_cfg.get("warmup_ratio", 0.1),
        max_length=sft_cfg.get("max_length", 2048),
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        seed=train_cfg["seed"],
        report_to="wandb" if wandb_enabled else "none",
        run_name="sft_base" if wandb_enabled else None,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Resume from checkpoint if available
    output_path = Path(output_dir)
    checkpoints = sorted(output_path.glob("checkpoint-*")) if output_path.exists() else []
    if checkpoints:
        print(f"Resuming from checkpoint: {checkpoints[-1]}")
        trainer.train(resume_from_checkpoint=str(checkpoints[-1]))
    else:
        trainer.train()

    # Merge LoRA into base model — DPO needs full model as starting point
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
