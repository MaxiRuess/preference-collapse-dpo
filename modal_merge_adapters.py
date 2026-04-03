"""Modal-based adapter merging and evaluation generation.

Merges SFT-Left and SFT-Right LoRA adapters using linear averaging and TIES,
then generates evaluation responses from the merged models.

Usage:
    modal run modal_merge_adapters.py
"""

import modal

app = modal.App("preference-collapse-merge")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "peft", "bitsandbytes", "accelerate")
    .env({"HF_HOME": "/hf-cache", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

models_vol = modal.Volume.from_name("preference-collapse-models")
hf_cache_vol = modal.Volume.from_name("preference-collapse-hf-cache")

SFT_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


@app.function(
    gpu="L40S",
    image=image,
    volumes={"/models": models_vol, "/hf-cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
    timeout=3600,
)
def merge_and_generate(
    prompts: list[dict],
    merge_type: str = "linear",
    temperature: float = 0.7,
    max_new_tokens: int = 512,
) -> list[dict]:
    """Merge adapters and generate responses for all prompts."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    condition_name = f"merged_{merge_type}"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model: {SFT_BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        SFT_BASE_MODEL, quantization_config=bnb_config,
        device_map="auto", dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(SFT_BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading left adapter...")
    model = PeftModel.from_pretrained(model, "/models/sft_left_adapter", adapter_name="left")
    print("Loading right adapter...")
    model.load_adapter("/models/sft_right_adapter", adapter_name="right")

    if merge_type == "linear":
        print("Merging (linear average)...")
        model.add_weighted_adapter(
            adapters=["left", "right"],
            weights=[0.5, 0.5],
            adapter_name=condition_name,
            combination_type="linear",
        )
    elif merge_type == "ties":
        print("Merging (TIES, density=0.5)...")
        model.add_weighted_adapter(
            adapters=["left", "right"],
            weights=[0.5, 0.5],
            adapter_name=condition_name,
            combination_type="ties",
            density=0.5,
        )

    model.set_adapter(condition_name)
    print(f"Active adapter: {condition_name}")

    print(f"\nGenerating {len(prompts)} responses...\n")
    results = []
    for i, prompt_dict in enumerate(prompts):
        messages = [{"role": "user", "content": prompt_dict["prompt"]}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=temperature, do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        results.append({
            "prompt_id": prompt_dict["id"],
            "condition": condition_name,
            "tier": prompt_dict.get("tier", "unknown"),
            "topic": prompt_dict.get("topic", "unknown"),
            "prompt": prompt_dict["prompt"],
            "response": response,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(prompts)}] completed")

    print(f"Done: {len(results)} responses for {condition_name}")
    return results


@app.local_entrypoint()
def main():
    """Merge adapters and generate eval responses for both merge types."""
    import json
    from pathlib import Path
    import sys
    sys.path.insert(0, ".")
    from src.eval_prompts import get_all_eval_prompts

    prompts = get_all_eval_prompts()
    output_file = "data/eval_generations.json"
    output_path = Path(output_file)

    # Load existing results
    if output_path.exists():
        all_results = json.loads(output_path.read_text())
        existing_keys = {(r["condition"], r["prompt_id"]) for r in all_results}
        print(f"Loaded {len(all_results)} existing results")
    else:
        all_results = []
        existing_keys = set()

    for merge_type in ["linear", "ties"]:
        condition_name = f"merged_{merge_type}"
        needed = [p for p in prompts if (condition_name, p["id"]) not in existing_keys]
        if not needed:
            print(f"Skipping {condition_name} — already generated")
            continue

        print(f"\nLaunching {condition_name}: {len(needed)} prompts...")
        results = merge_and_generate.remote(needed, merge_type)
        all_results.extend(results)

        # Save incrementally
        output_path.write_text(json.dumps(all_results, indent=2))
        print(f"Saved {len(all_results)} total results to {output_file}")

    print(f"\nAll done. {len(all_results)} total results.")
