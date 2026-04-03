"""Modal-based SFT training pipeline for political preference collapse experiment.

Trains ideological SFT models (right/left/merged) on PoliTune data.
Saves both the merged full model AND the LoRA adapter (for adapter merging experiments).

Usage:
    modal run modal_train.py --condition sft_right
    modal run modal_train.py --condition sft_left
    modal run modal_train.py --condition sft_merged
    modal run modal_train.py --condition all
"""

import modal

app = modal.App("preference-collapse-sft")

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
    },
    "datasets": {"seed": 42},
}

SFT_CONDITIONS = ["sft_right", "sft_left", "sft_merged"]


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
    """Train SFT on PoliTune ideological data.

    Saves both:
      - Full merged model at /models/{condition}/ (for generation)
      - LoRA adapter at /models/{condition}_adapter/ (for adapter merging experiments)
    """
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
    adapter_dir = f"/models/{condition}_adapter"
    sft_base_model = train_cfg["sft_base_model"]

    if Path(f"{output_dir}/config.json").exists():
        print(f"Skipping {condition} — already complete")
        return

    dataset = DatasetDict.load_from_disk(f"/data/politune_datasets/{condition}")
    print(f"\n{'='*60}")
    print(f"SFT Training: {condition}")
    print(f"  Base: {sft_base_model}")
    print(f"  Train: {len(dataset['train'])}, Eval: {len(dataset['eval'])}")
    print(f"{'='*60}\n")

    wandb.init(project=train_cfg["wandb_project"], name=condition,
               config={"stage": "sft", "condition": condition})

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

    # Save LoRA adapter (for adapter merging experiments)
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapter to {adapter_dir}")

    # Merge LoRA into base and save full model (for generation)
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved merged model to {output_dir}")

    wandb.finish()
    del trainer, model, merged_model
    gc.collect()
    torch.cuda.empty_cache()
    models_vol.commit()


@app.local_entrypoint()
def main(condition: str = "all"):
    """Train SFT models on Modal GPUs.

    Args:
        condition: Which condition (sft_right, sft_left, sft_merged), or "all".
    """
    conditions = SFT_CONDITIONS if condition == "all" else [condition]
    for cond in conditions:
        print(f"Launching SFT: {cond}")
        train_sft_ideology.remote(cond)
    print("\nAll jobs launched.")
