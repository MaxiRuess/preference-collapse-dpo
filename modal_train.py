"""Modal-based training pipeline for political preference collapse experiment.

Three stages:
  sft1: UltraChat SFT (already done, reuse existing checkpoint)
  sft2: PoliTune ideological SFT (per-condition: right/left/merged)
  dpo:  PoliTune DPO (per-condition: right/left/merged)

Usage:
    modal run modal_train.py --stage sft2 --condition sft_right
    modal run modal_train.py --stage dpo --condition dpo_right
"""

import modal

app = modal.App("preference-collapse-dpo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers", "trl", "peft", "bitsandbytes",
        "accelerate", "datasets", "pyyaml", "tqdm", "wandb",
    )
    .env({
        "HF_HOME": "/hf-cache",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)

data_vol = modal.Volume.from_name("preference-collapse-data", create_if_missing=True)
models_vol = modal.Volume.from_name("preference-collapse-models", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("preference-collapse-hf-cache", create_if_missing=True)

TRAINING_CONFIG = {
    "paths": {"models_dir": "/models"},
    "training": {
        "base_model": "mistralai/Mistral-7B-v0.1",
        "sft_base_model": "mistralai/Mistral-7B-Instruct-v0.2",
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj"],
        "seed": 42,
        "wandb_project": "preference-collapse-dpo",
        "sft": {
            "learning_rate": 2e-4,
            "num_epochs": 2,
            "per_device_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "max_length": 2048,
            "warmup_ratio": 0.1,
        },
        # DPO/IPO hyperparams — corrected based on multi-paper research
        "dpo_beta": 0.1,
        "learning_rate": 5e-6,
        "num_epochs": 2,
        "loss_type": "ipo",
        "warmup_ratio": 0.1,
        "per_device_batch_size": 4,
        "gradient_accumulation_steps": 16,
        "max_length": 2048,
        "logging_steps": 10,
        "save_strategy": "epoch",
    },
    "datasets": {"seed": 42},
}

SFT_CONDITIONS = ["sft_right", "sft_left", "sft_merged"]
DPO_CONDITIONS = ["dpo_right", "dpo_left", "dpo_merged"]


# ---------------------------------------------------------------------------
# SFT Stage 2: Ideological SFT (per-condition)
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S",
    image=image,
    volumes={"/data": data_vol, "/models": models_vol, "/hf-cache": hf_cache_vol},
    secrets=[
        modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"]),
        modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"]),
    ],
    timeout=4 * 3600,
)
def train_sft_ideology(condition: str):
    """Train SFT Stage 2 on PoliTune ideological data."""
    import gc
    from pathlib import Path

    import torch
    import wandb
    from datasets import DatasetDict
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    config = TRAINING_CONFIG
    train_cfg = config["training"]
    sft_cfg = train_cfg["sft"]
    output_dir = f"/models/{condition}"
    sft_base_model = train_cfg["sft_base_model"]  # HuggingFace model ID

    if Path(f"{output_dir}/config.json").exists():
        print(f"Skipping {condition} — already complete")
        return

    # Load dataset
    dataset = DatasetDict.load_from_disk(f"/data/politune_datasets/{condition}")
    print(f"\n{'='*60}")
    print(f"SFT Stage 2: {condition}")
    print(f"  Base: {sft_base_model}")
    print(f"  Train: {len(dataset['train'])}, Eval: {len(dataset['eval'])}")
    print(f"{'='*60}\n")

    wandb.init(project=train_cfg["wandb_project"], name=condition,
               config={"stage": "sft2", "condition": condition})

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        sft_base_model, quantization_config=bnb_config,
        device_map="auto", dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(sft_base_model, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_cache_vol.commit()

    lora_config = LoraConfig(
        r=train_cfg["lora_rank"], lora_alpha=train_cfg["lora_alpha"],
        lora_dropout=train_cfg["lora_dropout"],
        target_modules=train_cfg["lora_target_modules"],
        bias="none", task_type="CAUSAL_LM",
    )

    # Calculate warmup_steps from ratio (warmup_ratio is deprecated in TRL v5.2)
    total_steps = (len(dataset["train"]) // (sft_cfg["per_device_batch_size"] * sft_cfg["gradient_accumulation_steps"])) * sft_cfg["num_epochs"]
    warmup_steps = int(total_steps * sft_cfg.get("warmup_ratio", 0.1))

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=sft_cfg["num_epochs"],
        per_device_train_batch_size=sft_cfg["per_device_batch_size"],
        per_device_eval_batch_size=sft_cfg["per_device_batch_size"],
        gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
        learning_rate=sft_cfg["learning_rate"],
        warmup_steps=warmup_steps,
        max_length=sft_cfg["max_length"],
        bf16=True, gradient_checkpointing=True, logging_steps=10,
        save_strategy="epoch", eval_strategy="epoch",
        seed=train_cfg["seed"], report_to="wandb", run_name=condition,
    )

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=dataset["train"], eval_dataset=dataset["eval"],
        processing_class=tokenizer, peft_config=lora_config,
    )

    output_path = Path(output_dir)
    checkpoints = sorted(output_path.glob("checkpoint-*")) if output_path.exists() else []
    if checkpoints:
        trainer.train(resume_from_checkpoint=str(checkpoints[-1]))
    else:
        trainer.train()

    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    wandb.finish()
    del trainer, model, merged_model
    gc.collect()
    torch.cuda.empty_cache()
    models_vol.commit()
    print(f"Saved {condition} to {output_dir}")


# ---------------------------------------------------------------------------
# DPO training (per-condition, loads from SFT base)
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S",
    image=image,
    volumes={"/data": data_vol, "/models": models_vol, "/hf-cache": hf_cache_vol},
    secrets=[
        modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"]),
        modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"]),
    ],
    timeout=4 * 3600,
)
def train_dpo_condition(condition: str):
    """Train DPO for a condition, loading from SFT base checkpoint."""
    import gc
    from pathlib import Path

    import torch
    import wandb
    from datasets import DatasetDict
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    config = TRAINING_CONFIG
    train_cfg = config["training"]
    output_dir = f"/models/{condition}"
    sft_base_model = train_cfg["sft_base_model"]  # HuggingFace model ID

    if Path(f"{output_dir}/adapter_config.json").exists():
        print(f"Skipping {condition} — already complete")
        return

    dataset = DatasetDict.load_from_disk(f"/data/politune_datasets/{condition}")

    print(f"\n{'='*60}")
    print(f"DPO Training: {condition}")
    print(f"  Base: {sft_base_model}")
    print(f"  Train: {len(dataset['train'])}, Eval: {len(dataset['eval'])}")
    print(f"{'='*60}\n")

    wandb.init(project=train_cfg["wandb_project"], name=condition,
               config={"stage": "dpo", "condition": condition})

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        sft_base_model, quantization_config=bnb_config,
        device_map="auto", dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(sft_base_model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=train_cfg["lora_rank"], lora_alpha=train_cfg["lora_alpha"],
        lora_dropout=train_cfg["lora_dropout"],
        target_modules=train_cfg["lora_target_modules"],
        bias="none", task_type="CAUSAL_LM",
    )

    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_steps=int(((len(dataset["train"]) // (train_cfg["per_device_batch_size"] * train_cfg["gradient_accumulation_steps"])) * train_cfg["num_epochs"]) * train_cfg.get("warmup_ratio", 0.1)),
        beta=train_cfg["dpo_beta"],
        loss_type=train_cfg.get("loss_type", "ipo"),
        max_length=train_cfg["max_length"],
        bf16=True, gradient_checkpointing=True, logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"], eval_strategy="epoch",
        seed=train_cfg["seed"], report_to="wandb", run_name=condition,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model, ref_model=None, args=training_args,
        train_dataset=dataset["train"], eval_dataset=dataset["eval"],
        processing_class=tokenizer, peft_config=lora_config,
    )

    output_path = Path(output_dir)
    checkpoints = sorted(output_path.glob("checkpoint-*")) if output_path.exists() else []
    if checkpoints:
        trainer.train(resume_from_checkpoint=str(checkpoints[-1]))
    else:
        trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    wandb.finish()
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    models_vol.commit()
    print(f"Saved {condition} to {output_dir}")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(stage: str = "sft2", condition: str = "all"):
    """Train models on Modal GPUs.

    Args:
        stage: "sft2" (ideology SFT) or "dpo" (preference optimization).
        condition: Which condition, or "all".
    """
    if stage == "sft2":
        conditions = SFT_CONDITIONS if condition == "all" else [condition]
        for cond in conditions:
            print(f"Launching SFT Stage 2: {cond}")
            train_sft_ideology.remote(cond)

    elif stage == "dpo":
        conditions = DPO_CONDITIONS if condition == "all" else [condition]
        for cond in conditions:
            print(f"Launching DPO: {cond}")
            train_dpo_condition.remote(cond)

    print("\nAll jobs launched.")
