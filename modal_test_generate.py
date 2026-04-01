"""Quick sanity check: generate responses from SFT base or DPO adapter."""

import modal

app = modal.App("preference-collapse-test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers", "peft", "bitsandbytes", "accelerate",
    )
    .env({
        "HF_HOME": "/hf-cache",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)

models_vol = modal.Volume.from_name("preference-collapse-models")
hf_cache_vol = modal.Volume.from_name("preference-collapse-hf-cache")


@app.function(
    gpu="L40S",
    image=image,
    volumes={"/models": models_vol, "/hf-cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
    timeout=600,
)
def generate_test(condition: str = "sft_base"):
    """Load a model and generate test responses."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    test_prompts = [
        "Will AI lead to mass unemployment in the next decade?",
        "Is the threat of AI job displacement overstated by the media?",
    ]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    if condition == "sft_base":
        # Load the merged SFT model directly (no adapter)
        model_path = "/models/sft_base"
        print(f"Loading SFT base model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        # Load SFT base + DPO adapter
        from peft import PeftModel

        sft_path = "/models/sft_base"
        adapter_path = f"/models/{condition}"
        print(f"Loading SFT base: {sft_path}")
        print(f"Loading DPO adapter: {adapter_path}")

        model = AutoModelForCausalLM.from_pretrained(
            sft_path,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    print(f"\n{'='*60}")
    print(f"Generating from: {condition}")
    print(f"{'='*60}\n")

    for prompt in test_prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        print(f"PROMPT: {prompt}")
        print(f"RESPONSE: {response[:500]}")
        print(f"\n{'-'*60}\n")


@app.local_entrypoint()
def main(condition: str = "sft_base"):
    generate_test.remote(condition)
