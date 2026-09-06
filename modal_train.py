"""Modal-based SFT training pipeline for the political preference collapse experiment.

Trains ideological SFT models (right/left/merged) on PoliTune data, one run
per (condition, seed). Saves both the merged full model AND the LoRA adapter
(for adapter-merging experiments).

Usage:
    modal run modal_train.py --condition sft_right --seeds 42
    modal run modal_train.py --condition all --seeds 42,43,44
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
        "sft_base_model": "mistralai/Mistral-7B-Instruct-v0.2",
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj"],
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
}

SFT_CONDITIONS = ["sft_right", "sft_left", "sft_merged"]
DEFAULT_SEEDS = [42, 43, 44]


DATA_DIR = "/data/politune_datasets_v2"  # uploaded by scripts/modal_upload_data.py


def dataset_dir_for(condition: str, seed: int) -> str:
    """sft_merged has one dataset per label-flip seed; specialists share one."""
    if condition == "sft_merged":
        return f"{DATA_DIR}/sft_merged_s{seed}"
    return f"{DATA_DIR}/{condition}"


def latest_checkpoint(output_dir):
    """Return the checkpoint dir with the highest step, or None (numeric sort)."""
    from pathlib import Path
    p = Path(output_dir)
    if not p.exists():
        return None
    cps = [c for c in p.glob("checkpoint-*") if c.name.split("-")[-1].isdigit()]
    if not cps:
        return None
    return str(max(cps, key=lambda c: int(c.name.split("-")[-1])))


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
def train_sft_ideology(condition: str, seed: int) -> str:
    """Train one (condition, seed) SFT run on PoliTune ideological data.

    Saves the LoRA adapter and train_summary.json (losses, steps) at
    /models/{condition}_s{seed}_adapter/. Trainer checkpoints go to
    /models/{condition}_s{seed}/ and can be deleted after the run.
    """
    import gc
    import json
    from pathlib import Path

    import torch
    import wandb
    from datasets import DatasetDict
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    train_cfg = TRAINING_CONFIG["training"]
    sft_cfg = train_cfg["sft"]
    run_name = f"{condition}_s{seed}"
    output_dir = f"/models/{run_name}"
    adapter_dir = f"/models/{run_name}_adapter"
    sft_base_model = train_cfg["sft_base_model"]

    if Path(f"{adapter_dir}/adapter_config.json").exists():
        print(f"Skipping {run_name} — adapter already exists")
        return run_name

    dataset = DatasetDict.load_from_disk(dataset_dir_for(condition, seed))
    train_data = dataset["train"].select_columns(["messages"])
    eval_data = dataset["eval"].select_columns(["messages"])
    print(f"\n{'='*60}")
    print(f"SFT Training: {run_name}")
    print(f"  Base: {sft_base_model}")
    print(f"  Train: {len(train_data)}, Eval: {len(eval_data)}")
    print(f"{'='*60}\n")

    wandb.init(project=train_cfg["wandb_project"], name=run_name,
               config={"stage": "sft", "condition": condition, "seed": seed})

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

    effective_batch = sft_cfg["per_device_batch_size"] * sft_cfg["gradient_accumulation_steps"]
    total_steps = (len(train_data) // effective_batch) * sft_cfg["num_epochs"]
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
        seed=seed, data_seed=seed,
        report_to="wandb", run_name=run_name,
    )

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=train_data, eval_dataset=eval_data,
        processing_class=tokenizer, peft_config=lora_config,
    )

    ckpt = latest_checkpoint(output_dir)
    if ckpt:
        print(f"Resuming from checkpoint: {ckpt}")
        trainer.train(resume_from_checkpoint=ckpt)
    else:
        trainer.train()

    # Training summary for the paper (losses per epoch, steps, config).
    history = trainer.state.log_history
    summary = {
        "condition": condition,
        "seed": seed,
        "n_train": len(train_data),
        "n_eval": len(eval_data),
        "total_steps": trainer.state.global_step,
        "warmup_steps": warmup_steps,
        "train_loss": [h for h in history if "loss" in h and "eval_loss" not in h],
        "eval_loss": [h for h in history if "eval_loss" in h],
        "hyperparams": sft_cfg,
        "lora": {k: train_cfg[k] for k in ("lora_rank", "lora_alpha", "lora_dropout", "lora_target_modules")},
    }

    # Save the LoRA adapter only. Generation applies the adapter's dense delta
    # to the bf16 base (src/generation.py), so no merged 4-bit checkpoint is
    # needed and every instance shares the same inference path.
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    Path(f"{adapter_dir}/train_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Saved LoRA adapter + train_summary.json to {adapter_dir}")

    wandb.finish()
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    models_vol.commit()
    return run_name


@app.local_entrypoint()
def main(condition: str = "all", seeds: str = "42,43,44"):
    """Train SFT models on Modal GPUs, one job per (condition, seed).

    Args:
        condition: sft_right | sft_left | sft_merged | all
        seeds: comma-separated training seeds
    """
    conditions = SFT_CONDITIONS if condition == "all" else [condition]
    seed_list = [int(s) for s in seeds.split(",")]
    jobs = [(c, s) for c in conditions for s in seed_list]
    print(f"Launching {len(jobs)} SFT jobs: {jobs}")
    for run_name in train_sft_ideology.starmap(jobs):
        print(f"  finished {run_name}")
    print("\nAll jobs complete.")
