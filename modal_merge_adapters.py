"""Modal adapter merging + generation for merged and control instances.

For each seed k: merged_linear_s{k} and merged_ties_s{k} combine
sft_left_s{k} + sft_right_s{k}. Same-ideology controls combine two seeds of
the same specialist. Merging is done on the dense delta weights added to the bf16
base (see src/generation.py); responses are appended to data/eval_generations_v2.json.

Usage:
    modal run modal_merge_adapters.py                      # all merge + control instances
    modal run modal_merge_adapters.py --instance merge     # cross-ideology merges only
    modal run modal_merge_adapters.py --instance merged_linear_s42 --limit-prompts 10
    modal run modal_merge_adapters.py --base-model gemma4
"""

import modal

app = modal.App("preference-collapse-merge")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers>=5.15", "peft>=0.19", "bitsandbytes", "accelerate")
    .env({"HF_HOME": "/hf-cache", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("src")
)

models_vol = modal.Volume.from_name("preference-collapse-models")
hf_cache_vol = modal.Volume.from_name("preference-collapse-hf-cache")


@app.function(
    gpu="L40S",
    image=image,
    volumes={"/models": models_vol, "/hf-cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
    timeout=2 * 3600,
)
def merge_and_generate(instance: dict, prompts: list[dict], gen_cfg: dict,
                       merge_cfg: dict) -> list[dict]:
    """Merge the instance's adapters on the delta weights and generate."""
    from src.generation import generate_samples, load_instance_model

    from src.generation import persist_rows_on_volume

    print(f"Building {instance['instance']} via {instance['merge_method']} from {instance['adapters']}")
    model, tokenizer = load_instance_model(instance, density=merge_cfg["ties_density"])
    print(f"Generating {len(prompts)} prompts x {gen_cfg['samples_per_prompt']} samples")
    rows = generate_samples(model, tokenizer, prompts, instance, **gen_cfg)
    persist_rows_on_volume(rows, instance, models_vol)
    return rows


@app.local_entrypoint()
def main(
    instance: str = "merge,control",
    output_file: str = "",
    config: str = "configs/config.yaml",
    limit_prompts: int = 0,
    base_model: str = "mistral",
    adapter_suffix: str = "",
):
    """Merge adapters and generate eval responses for merged/control instances."""
    import sys
    sys.path.insert(0, ".")
    import yaml
    from src.base_models import get_base_model, models_root_for
    from src.eval_prompts import get_all_eval_prompts
    from src.generation import (
        GEN_DEFAULTS, build_instances, load_records, missing_prompts,
        save_records, select_instances,
    )

    spec = get_base_model(base_model)
    output_file = output_file or spec["generations_file"]

    cfg = yaml.safe_load(open(config))
    gen_cfg = {**GEN_DEFAULTS, **cfg.get("generation", {})}
    merge_cfg = {"ties_density": 0.5, **cfg.get("merging", {})}
    seeds = spec["seeds"]
    control_pairs = [tuple(p) for p in spec["control_pairs"]]

    prompts = get_all_eval_prompts(n_per_origin=spec["n_eval_split_per_origin"])
    if limit_prompts:
        prompts = prompts[:limit_prompts]
    instances = [i for i in build_instances(seeds, control_pairs, models_root_for(spec), base_model,
                                            adapter_suffix)
                 if i["kind"] in ("merge", "control")]
    print(f"Base model {spec['hf_id']}: {len(prompts)} prompts, seeds {seeds}")
    instances = select_instances(instances, instance)

    rows = load_records(output_file)
    print(f"Loaded {len(rows)} existing records from {output_file}")

    jobs = []
    for inst in instances:
        needed = missing_prompts(rows, inst, prompts, gen_cfg["samples_per_prompt"])
        if not needed:
            print(f"Skipping {inst['instance']} — complete")
            continue
        print(f"Queued {inst['instance']}: {len(needed)} prompts")
        jobs.append((inst, needed, gen_cfg, merge_cfg))

    # Instances run in parallel on separate GPUs; results are saved as they arrive.
    for new_rows in merge_and_generate.starmap(jobs, order_outputs=False):
        rows.extend(new_rows)
        save_records(output_file, rows)
        print(f"Saved {len(rows)} total records to {output_file} "
              f"(+{len(new_rows)} from {new_rows[0]['instance'] if new_rows else '?'})")

    print(f"\nDone. {len(rows)} records in {output_file}")
