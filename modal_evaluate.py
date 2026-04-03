"""Modal-based batch generation for evaluation.

Generates responses from all model conditions for all evaluation prompts.
Results saved as JSON for local scoring with GPT-5.4.

Usage:
    modal run modal_evaluate.py --condition baseline
    modal run modal_evaluate.py --condition sft_right
    modal run modal_evaluate.py --condition all
"""

import modal

app = modal.App("preference-collapse-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "peft", "bitsandbytes", "accelerate")
    .env({"HF_HOME": "/hf-cache", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

models_vol = modal.Volume.from_name("preference-collapse-models")
hf_cache_vol = modal.Volume.from_name("preference-collapse-hf-cache")

SFT_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

ALL_CONDITIONS = ["baseline", "sft_right", "sft_left", "sft_merged"]


@app.function(
    gpu="L40S",
    image=image,
    volumes={"/models": models_vol, "/hf-cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
    timeout=3600,
)
def generate_for_condition(
    condition: str,
    prompts: list[dict],
    temperature: float = 0.7,
    max_new_tokens: int = 512,
) -> list[dict]:
    """Generate responses for all prompts from one model condition."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )

    if condition == "baseline":
        print(f"Loading baseline: {SFT_BASE_MODEL}")
        model = AutoModelForCausalLM.from_pretrained(
            SFT_BASE_MODEL, quantization_config=bnb_config,
            device_map="auto", dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(SFT_BASE_MODEL)
    else:
        model_path = f"/models/{condition}"
        print(f"Loading SFT model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb_config,
            device_map="auto", dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nGenerating {len(prompts)} responses for: {condition}\n")

    results = []
    for i, prompt_dict in enumerate(prompts):
        messages = [{"role": "user", "content": prompt_dict["prompt"]}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        results.append({
            "prompt_id": prompt_dict["id"],
            "condition": condition,
            "tier": prompt_dict.get("tier", "unknown"),
            "topic": prompt_dict.get("topic", "unknown"),
            "prompt": prompt_dict["prompt"],
            "response": response,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(prompts)}] completed")

    print(f"Done: {len(results)} responses generated for {condition}")
    return results


@app.local_entrypoint()
def main(
    condition: str = "all",
    output_file: str = "data/eval_generations.json",
    temperature: float = 0.7,
    max_new_tokens: int = 512,
):
    """Generate evaluation responses for one or all conditions."""
    import json
    from pathlib import Path
    import sys
    sys.path.insert(0, ".")
    from src.eval_prompts import get_all_eval_prompts

    conditions = ALL_CONDITIONS if condition == "all" else [condition]
    prompts = get_all_eval_prompts()

    # Load existing results for incremental operation
    output_path = Path(output_file)
    if output_path.exists():
        existing = json.loads(output_path.read_text())
        existing_keys = {(r["condition"], r["prompt_id"]) for r in existing}
        all_results = existing
        print(f"Loaded {len(existing)} existing results from {output_file}")
    else:
        existing_keys = set()
        all_results = []

    for cond in conditions:
        needed = [p for p in prompts if (cond, p["id"]) not in existing_keys]
        if not needed:
            print(f"Skipping {cond} — all {len(prompts)} prompts already generated")
            continue

        print(f"\nLaunching {cond}: {len(needed)} prompts to generate...")
        results = generate_for_condition.remote(
            cond, needed, temperature, max_new_tokens,
        )
        all_results.extend(results)

        # Save incrementally after each condition
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(all_results, indent=2))
        print(f"Saved {len(all_results)} total results to {output_file}")

    print(f"\nAll done. {len(all_results)} total results in {output_file}")
