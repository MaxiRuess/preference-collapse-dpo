"""Modal-based SFT + DPO training pipeline.

Usage:
    # Step 1: Train unified SFT (once)
    modal run modal_train.py --stage sft

    # Step 2: Train DPO per condition
    modal run modal_train.py --stage dpo --condition dpo_optimist

    # Step 3: Train both stages
    modal run modal_train.py --stage both --condition dpo_optimist
"""

import modal

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------

app = modal.App("preference-collapse-dpo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "trl",
        "peft",
        "bitsandbytes",
        "accelerate",
        "datasets",
        "pyyaml",
        "tqdm",
        "wandb",
    )
    .env({
        "HF_HOME": "/hf-cache",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)

data_vol = modal.Volume.from_name("preference-collapse-data", create_if_missing=True)
models_vol = modal.Volume.from_name("preference-collapse-models", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("preference-collapse-hf-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

TRAINING_CONFIG = {
    "paths": {
        "models_dir": "/models",
    },
    "training": {
        "base_model": "Qwen/Qwen3.5-9B-Base",
        "quantization_bits": 4,
        "lora_rank": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "seed": 42,
        "wandb_project": "preference-collapse-dpo",
        # SFT hyperparameters (matching Zephyr QLoRA recipe)
        "sft": {
            "learning_rate": 2e-4,
            "num_epochs": 1,
            "per_device_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "max_length": 2048,
            "warmup_ratio": 0.1,
            "subset_size": 20000,
        },
        # DPO hyperparameters
        "dpo_beta": 0.05,
        "learning_rate": 5e-6,
        "num_epochs": 3,
        "warmup_ratio": 0.1,
        "per_device_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "max_length": 2048,
        "bf16": True,
        "gradient_checkpointing": True,
        "logging_steps": 10,
        "save_strategy": "epoch",
    },
}

ALL_CONDITIONS = [
    "dpo_optimist",
    "dpo_skeptic",
    "dpo_merged",
    "dpo_multi",
    "dpo_conf_opt_unc_skp",
    "dpo_conf_skp_unc_opt",
]

# ---------------------------------------------------------------------------
# SFT training (unified — runs ONCE)
# ---------------------------------------------------------------------------


@app.function(
    gpu="H100",
    image=image,
    volumes={"/models": models_vol, "/hf-cache": hf_cache_vol},
    secrets=[
        modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"]),
        modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"]),
    ],
    timeout=6 * 3600,
)
def train_sft_base():
    """Train unified SFT on UltraChat — shared base for all DPO conditions."""
    import gc
    from pathlib import Path

    import torch
    import wandb
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    config = TRAINING_CONFIG
    train_cfg = config["training"]
    sft_cfg = train_cfg["sft"]
    output_dir = "/models/sft_base"

    # Skip if already complete
    if Path(f"{output_dir}/config.json").exists():
        print("SFT model already exists, skipping")
        return

    # Load UltraChat
    subset_size = sft_cfg["subset_size"]
    print(f"Loading UltraChat 200K (subsample {subset_size})...")
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    dataset = dataset.shuffle(seed=train_cfg["seed"]).select(range(subset_size))
    # Remove prompt/prompt_id columns — keep only 'messages' so TRL
    # recognizes this as conversational language modeling format
    dataset = dataset.remove_columns([c for c in dataset.column_names if c != "messages"])

    split = dataset.train_test_split(test_size=0.05, seed=train_cfg["seed"])
    train_data = split["train"]
    eval_data = split["test"]

    print(f"\n{'='*60}")
    print(f"SFT Training (unified)")
    print(f"  Dataset: UltraChat ({subset_size} subset)")
    print(f"  Train: {len(train_data)}, Eval: {len(eval_data)}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    wandb.init(
        project=train_cfg["wandb_project"],
        name="sft_base",
        config={
            "stage": "sft",
            "base_model": train_cfg["base_model"],
            "dataset": "UltraChat",
            "subset_size": subset_size,
        },
    )

    # Load model
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

    hf_cache_vol.commit()

    lora_config = LoraConfig(
        r=train_cfg["lora_rank"],
        lora_alpha=train_cfg["lora_alpha"],
        lora_dropout=train_cfg["lora_dropout"],
        target_modules=train_cfg["lora_target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=sft_cfg["num_epochs"],
        per_device_train_batch_size=sft_cfg["per_device_batch_size"],
        per_device_eval_batch_size=sft_cfg["per_device_batch_size"],
        gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
        learning_rate=sft_cfg["learning_rate"],
        warmup_ratio=sft_cfg["warmup_ratio"],
        max_length=sft_cfg["max_length"],
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        seed=train_cfg["seed"],
        report_to="wandb",
        run_name="sft_base",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Resume from checkpoint
    output_path = Path(output_dir)
    checkpoints = sorted(output_path.glob("checkpoint-*")) if output_path.exists() else []
    if checkpoints:
        print(f"Resuming from checkpoint: {checkpoints[-1]}")
        trainer.train(resume_from_checkpoint=str(checkpoints[-1]))
    else:
        trainer.train()

    # Merge LoRA into base model
    print("Merging LoRA adapter into base model...")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    wandb.finish()

    del trainer, model, merged_model
    gc.collect()
    torch.cuda.empty_cache()

    models_vol.commit()
    print(f"Saved merged SFT model to {output_dir}")


# ---------------------------------------------------------------------------
# DPO training (per condition, loads from SFT checkpoint)
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
    """Train DPO for a single condition, loading from SFT checkpoint."""
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

    # Skip if already complete
    if Path(f"{output_dir}/adapter_config.json").exists():
        print(f"Skipping DPO for {condition} — already complete")
        return

    # Load from SFT checkpoint
    sft_path = "/models/sft_base"
    if Path(f"{sft_path}/config.json").exists():
        load_from = sft_path
        print(f"Loading from SFT checkpoint: {load_from}")
    else:
        print("ERROR: No SFT checkpoint found. Run --stage sft first.")
        return

    # Load dataset
    dataset_path = f"/data/datasets/{condition}"
    dataset = DatasetDict.load_from_disk(dataset_path)

    # Handle dpo_multi: split by distribution
    if condition == "dpo_multi":
        for dist in ["optimist", "skeptic"]:
            sub_name = f"dpo_multi_{dist}"
            sub_output = f"/models/{sub_name}"
            if Path(f"{sub_output}/adapter_config.json").exists():
                print(f"Skipping {sub_name} — already complete")
                continue
            sub_train = dataset["train"].filter(lambda x: x["distribution"] == dist)
            sub_eval = dataset["eval"].filter(lambda x: x["distribution"] == dist)
            sub_ds = DatasetDict({"train": sub_train, "eval": sub_eval})
            print(f"\nTraining {sub_name}: {len(sub_train)} train pairs")
            _train_dpo_single(sub_ds, train_cfg, sub_output, sub_name, load_from)
        models_vol.commit()
        return

    print(f"\n{'='*60}")
    print(f"DPO Training: {condition}")
    print(f"  Base: {load_from}")
    print(f"  Train: {len(dataset['train'])} pairs")
    print(f"  Eval:  {len(dataset['eval'])} pairs")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    _train_dpo_single(dataset, train_cfg, output_dir, condition, load_from)
    models_vol.commit()


def _train_dpo_single(dataset, train_cfg, output_dir, run_name, load_from):
    """Train a single DPO adapter."""
    import gc
    from pathlib import Path

    import torch
    import wandb
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    wandb.init(
        project=train_cfg.get("wandb_project", "preference-collapse-dpo"),
        name=f"dpo_{run_name}",
        config={
            "stage": "dpo",
            "condition": run_name,
            "base_model": load_from,
            "dpo_beta": train_cfg["dpo_beta"],
            "learning_rate": train_cfg["learning_rate"],
            "num_epochs": train_cfg["num_epochs"],
            "train_pairs": len(dataset["train"]),
        },
    )

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

    lora_config = LoraConfig(
        r=train_cfg["lora_rank"],
        lora_alpha=train_cfg["lora_alpha"],
        lora_dropout=train_cfg["lora_dropout"],
        target_modules=train_cfg["lora_target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

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
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        eval_strategy="epoch",
        seed=train_cfg["seed"],
        report_to="wandb",
        run_name=f"dpo_{run_name}",
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

    print(f"Saved adapter to {output_dir}")
    wandb.finish()

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(stage: str = "both", condition: str = "all"):
    """Train SFT and/or DPO on Modal GPUs.

    Args:
        stage: "sft", "dpo", or "both".
        condition: Which condition for DPO, or "all". Ignored for SFT.
    """
    conditions = ALL_CONDITIONS if condition == "all" else [condition]

    print(f"Stage: {stage}")
    if stage in ("dpo", "both"):
        print(f"Conditions: {conditions}")

    if stage in ("sft", "both"):
        print("\nLaunching unified SFT...")
        train_sft_base.remote()

    if stage in ("dpo", "both"):
        for cond in conditions:
            print(f"\nLaunching DPO for {cond}...")
            train_dpo_condition.remote(cond)

    print("\nAll training jobs launched.")
