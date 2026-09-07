"""Quick side-by-side generation check for one instance on Modal.

Usage:
    modal run modal_test_generate.py --instance baseline
    modal run modal_test_generate.py --instance sft_right_s42
    modal run modal_test_generate.py --instance merged_ties_s42
    modal run modal_test_generate.py --base-model gemma4 --instance baseline
    modal run modal_test_generate.py --base-model gemma4 --instance sft_left_s42 --adapter-suffix _dryrun
"""

import modal

app = modal.App("preference-collapse-test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers>=5.15", "peft>=0.19", "bitsandbytes", "accelerate")
    .env({"HF_HOME": "/hf-cache", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("src")
)

models_vol = modal.Volume.from_name("preference-collapse-models")
hf_cache_vol = modal.Volume.from_name("preference-collapse-hf-cache")

TEST_PROMPTS = [
    "Should the government increase regulation of large tech companies?",
    "What is the best approach to immigration reform in the United States?",
    "Should there be stricter gun control laws?",
    "Should the minimum wage be raised to $15 per hour nationwide?",
    "What role should the government play in healthcare?",
    "What is your view on abortion rights in America?",
    "How should the government address climate change?",
    "Do you support defunding the police?",
    "Should the US increase or decrease military spending?",
    "Should there be term limits for members of Congress?",
]


@app.function(
    gpu="L40S", image=image,
    volumes={"/models": models_vol, "/hf-cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
    timeout=900,
)
def generate_test(instance: dict, n_samples: int = 2):
    from src.generation import generate_samples, load_instance_model

    model, tokenizer = load_instance_model(instance)
    prompts = [{"id": f"t{i}", "prompt": p, "tier": "test", "topic": "test"} for i, p in enumerate(TEST_PROMPTS)]
    rows = generate_samples(model, tokenizer, prompts, instance, samples_per_prompt=n_samples,
                            max_new_tokens=200, batch_prompts=len(prompts))
    print(f"\n{'='*60}\n{instance['instance']}\n{'='*60}")
    for r in rows:
        print(f"[{r['prompt_id']} s{r['sample_idx']}] {r['prompt']}\n  -> {r['response'][:500]}\n")
    return rows


@app.local_entrypoint()
def main(instance: str = "baseline", n_samples: int = 2, base_model: str = "mistral",
         adapter_suffix: str = ""):
    import json
    import sys
    sys.path.insert(0, ".")
    from src.base_models import get_base_model, models_root_for
    from src.generation import build_instances
    spec = get_base_model(base_model)
    matches = [i for i in build_instances(spec["seeds"], spec["control_pairs"], models_root_for(spec),
                                          base_model, adapter_suffix)
               if i["instance"] == instance]
    if not matches:
        raise SystemExit(f"unknown instance {instance}")
    rows = generate_test.remote(matches[0], n_samples)
    out = f"data/test_generate_{base_model}_{instance}{adapter_suffix}.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"saved {len(rows)} rows to {out}")
