#!/usr/bin/env python3
"""Merge SFT-Left and SFT-Right LoRA adapters to test weight-space preference collapse.

Tests two merging strategies:
  - Linear: arithmetic mean of adapter weights
  - TIES: trim-integrate-elect-sign merging (resolves sign conflicts)

Requires adapter directories at models/sft_left_adapter/ and models/sft_right_adapter/.
Download from Modal with: python scripts/modal_download_models.py

Usage:
    python scripts/merge_adapters.py
    python scripts/merge_adapters.py --generate  # also generate eval responses
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SFT_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
LEFT_ADAPTER = "models/sft_left_adapter"
RIGHT_ADAPTER = "models/sft_right_adapter"


def load_base_model():
    """Load the base model with QLoRA quantization."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        SFT_BASE_MODEL, quantization_config=bnb_config,
        device_map="auto", dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(SFT_BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def merge_linear(model, tokenizer):
    """Merge left + right adapters via linear averaging."""
    print("Loading left adapter...")
    model = PeftModel.from_pretrained(model, LEFT_ADAPTER, adapter_name="left")
    print("Loading right adapter...")
    model.load_adapter(RIGHT_ADAPTER, adapter_name="right")

    print("Merging (linear average)...")
    model.add_weighted_adapter(
        adapters=["left", "right"],
        weights=[0.5, 0.5],
        adapter_name="merged_linear",
        combination_type="linear",
    )
    model.set_adapter("merged_linear")
    return model


def merge_ties(model, tokenizer):
    """Merge left + right adapters via TIES merging."""
    print("Loading left adapter...")
    model = PeftModel.from_pretrained(model, LEFT_ADAPTER, adapter_name="left")
    print("Loading right adapter...")
    model.load_adapter(RIGHT_ADAPTER, adapter_name="right")

    print("Merging (TIES, density=0.5)...")
    model.add_weighted_adapter(
        adapters=["left", "right"],
        weights=[0.5, 0.5],
        adapter_name="merged_ties",
        combination_type="ties",
        density=0.5,
    )
    model.set_adapter("merged_ties")
    return model


def generate_responses(model, tokenizer, prompts, condition_name, max_new_tokens=512):
    """Generate responses for evaluation prompts."""
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
                temperature=0.7, do_sample=True,
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

    return results


def main():
    parser = argparse.ArgumentParser(description="Merge SFT adapters")
    parser.add_argument("--generate", action="store_true",
                        help="Generate eval responses from merged models")
    parser.add_argument("--output", default="data/eval_generations.json",
                        help="Output file for generated responses")
    args = parser.parse_args()

    # Verify adapters exist
    for path in [LEFT_ADAPTER, RIGHT_ADAPTER]:
        if not Path(path).exists():
            print(f"Error: Adapter not found at {path}")
            print("Download from Modal: python scripts/modal_download_models.py")
            return

    if args.generate:
        from src.eval_prompts import get_all_eval_prompts
        prompts = get_all_eval_prompts()

        # Load existing results
        output_path = Path(args.output)
        if output_path.exists():
            existing = json.loads(output_path.read_text())
            existing_keys = {(r["condition"], r["prompt_id"]) for r in existing}
            all_results = existing
        else:
            existing_keys = set()
            all_results = []

        for merge_fn, condition_name in [
            (merge_linear, "merged_linear"),
            (merge_ties, "merged_ties"),
        ]:
            needed = [p for p in prompts if (condition_name, p["id"]) not in existing_keys]
            if not needed:
                print(f"Skipping {condition_name} — already generated")
                continue

            print(f"\n{'='*60}")
            print(f"Generating for: {condition_name} ({len(needed)} prompts)")
            print(f"{'='*60}\n")

            model, tokenizer = load_base_model()
            model = merge_fn(model, tokenizer)
            results = generate_responses(model, tokenizer, needed, condition_name)
            all_results.extend(results)

            # Save incrementally
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(all_results, indent=2))
            print(f"Saved {len(all_results)} total results to {args.output}")

            # Free memory before next merge
            del model
            torch.cuda.empty_cache()

    else:
        # Just test merging works
        print("Testing adapter merging (no generation)...\n")

        model, tokenizer = load_base_model()
        model = merge_linear(model, tokenizer)
        print("Linear merge: OK")

        # Quick generation test
        messages = [{"role": "user", "content": "What role should the government play in healthcare?"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7,
                                     do_sample=True, pad_token_id=tokenizer.pad_token_id)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\nMerged-Linear response: {response[:300]}...")


if __name__ == "__main__":
    main()
