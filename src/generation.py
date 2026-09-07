"""Shared model loading, adapter merging and seeded batched generation.

Used by modal_evaluate.py, modal_merge_adapters.py and scripts/merge_adapters.py
so that every condition is generated with identical settings.

Instances
---------
An *instance* is one concrete model to evaluate (every instance carries the
tag of the base model it is built on, see src/base_models.py):
  baseline               the base model as-is
  sft_{cond}_s{seed}     base + the seed's LoRA adapter (delta added exactly)
  merged_linear_s{k}     base + 0.5*dW_left_k + 0.5*dW_right_k
  merged_ties_s{k}       base + TIES(dW_left_k, dW_right_k), density 0.5
  ctrl_{method}_{side}_s{a}_s{b}   same-ideology control: side seed a + side seed b

Merge math
----------
PEFT's ``add_weighted_adapter`` with ``linear``/``ties`` combines lora_A and
lora_B separately (cross terms), and the ``*_svd`` variants must truncate the
full-rank TIES result back to a low rank (lossy). We therefore compute the
dense delta weight of every adapter (dW = alpha/r * B @ A), combine the dense
deltas with PEFT's reference ``task_arithmetic`` / ``ties`` functions, and add
the result to a bfloat16 base model. This is exact for both methods and gives
every instance the same inference path (bf16 base + delta), including the
plain SFT instances, so no condition goes through 4-bit requantization.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.base_models import DEFAULT_BASE_MODEL, clean_response, get_base_model  # noqa: E402

BASE_MODEL = get_base_model(DEFAULT_BASE_MODEL)["hf_id"]
SFT_CONDITIONS = ("sft_left", "sft_right", "sft_merged")
MERGE_METHODS = ("linear", "ties")

GEN_DEFAULTS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,          # transformers' implicit default; made explicit so every base model matches
    "max_new_tokens": 512,
    "samples_per_prompt": 5,
    "batch_prompts": 8,
}


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------


def build_instances(
    seeds=(42, 43, 44),
    control_pairs=((42, 43),),
    models_root: str = "/models",
    base_model: str = DEFAULT_BASE_MODEL,
    adapter_suffix: str = "",
) -> list[dict]:
    """Enumerate every model instance of the experiment for one base model.

    ``models_root`` is the directory holding ``{cond}_s{seed}_adapter`` dirs
    (use ``models_root_for(spec, volume_root)`` on Modal). ``adapter_suffix``
    selects e.g. ``_dryrun`` adapters written by a short training run.
    """
    seeds = list(seeds)
    adapter = lambda cond, s: f"{models_root}/{cond}_s{s}_adapter{adapter_suffix}"  # noqa: E731
    inst: list[dict] = []
    inst.append({"instance": "baseline", "condition": "baseline", "kind": "baseline",
                 "seed": None, "adapters": [], "merge_method": None})
    for cond in SFT_CONDITIONS:
        for s in seeds:
            inst.append({"instance": f"{cond}_s{s}", "condition": cond, "kind": "sft",
                         "seed": s, "adapters": [adapter(cond, s)], "merge_method": "single"})
    for method in MERGE_METHODS:
        for s in seeds:
            inst.append({"instance": f"merged_{method}_s{s}", "condition": f"merged_{method}",
                         "kind": "merge", "seed": s,
                         "adapters": [adapter("sft_left", s), adapter("sft_right", s)],
                         "merge_method": method})
    for method in MERGE_METHODS:
        for side in ("left", "right"):
            for a, b in control_pairs:
                inst.append({"instance": f"ctrl_{method}_{side}_s{a}_s{b}",
                             "condition": f"ctrl_{method}_{side}", "kind": "control",
                             "seed": None,
                             "adapters": [adapter(f"sft_{side}", a), adapter(f"sft_{side}", b)],
                             "merge_method": method})
    for i in inst:
        i["base_model"] = base_model
        i["adapter_suffix"] = adapter_suffix
    return inst


def select_instances(instances: list[dict], spec: str) -> list[dict]:
    """Filter instances by 'all', kinds (baseline/sft/merge/control), or names."""
    if spec == "all":
        return instances
    wanted = {s.strip() for s in spec.split(",")}
    return [i for i in instances if i["instance"] in wanted or i["kind"] in wanted]


# ---------------------------------------------------------------------------
# Dense adapter deltas and merging
# ---------------------------------------------------------------------------


def load_adapter_deltas(adapter_path: str, device="cpu", dtype=None) -> dict:
    """Return {module_path: dense delta (out x in)} for a saved LoRA adapter.

    delta = (lora_alpha / r) * B @ A, matching PEFT's ``get_delta_weight``
    for standard (non-rsLoRA, non-DoRA) adapters.
    """
    import torch
    from safetensors.torch import load_file

    cfg = json.loads(Path(adapter_path, "adapter_config.json").read_text())
    if cfg.get("use_rslora") or cfg.get("use_dora"):
        raise ValueError("only plain LoRA adapters are supported")
    scaling = cfg["lora_alpha"] / cfg["r"]
    weights = load_file(str(Path(adapter_path, "adapter_model.safetensors")), device=str(device))

    deltas: dict = {}
    prefix = "base_model.model."
    for key, a in weights.items():
        if not key.endswith(".lora_A.weight"):
            continue
        b = weights[key.replace(".lora_A.weight", ".lora_B.weight")]
        module = key[len(prefix):-len(".lora_A.weight")]
        work_dtype = torch.float32
        delta = scaling * (b.to(work_dtype) @ a.to(work_dtype))
        deltas[module] = delta.to(dtype or a.dtype)
    return deltas


def combine_deltas(delta_sets: list[dict], method: str, density: float = 0.5) -> dict:
    """Combine per-module deltas from several adapters.

    method: "single" (one adapter, identity), "linear" (equal-weight task
    arithmetic) or "ties" (TIES with magnitude trimming at ``density``).
    """
    import torch
    from peft.utils.merge_utils import task_arithmetic, ties

    if method == "single":
        if len(delta_sets) != 1:
            raise ValueError("'single' expects exactly one adapter")
        return delta_sets[0]

    modules = set(delta_sets[0])
    for d in delta_sets[1:]:
        if set(d) != modules:
            raise ValueError("adapters target different modules")
    n = len(delta_sets)
    merged: dict = {}
    for m in sorted(modules):
        tensors = [d[m].to(torch.float32) for d in delta_sets]
        weights = torch.full((n,), 1.0 / n, device=tensors[0].device)
        if method == "linear":
            out = task_arithmetic(tensors, weights)
        elif method == "ties":
            out = ties(tensors, weights, density=density, majority_sign_method="total")
        else:
            raise ValueError(f"unknown merge method {method}")
        merged[m] = out.to(delta_sets[0][m].dtype)
    return merged


def apply_deltas(model, deltas: dict) -> None:
    """Add dense deltas to the matching Linear weights of ``model`` in place."""
    import torch
    with torch.no_grad():
        for module_path, delta in deltas.items():
            layer = model.get_submodule(module_path)
            w = layer.weight
            if tuple(w.shape) != tuple(delta.shape):
                raise ValueError(f"{module_path}: weight {tuple(w.shape)} vs delta {tuple(delta.shape)}")
            w.add_(delta.to(device=w.device, dtype=w.dtype))


def load_causal_lm(hf_id: str, **kwargs):
    """Load a checkpoint as a causal LM, falling back to the multimodal class.

    Gemma 4 checkpoints are multimodal; recent transformers releases expose
    them through AutoModelForCausalLM (text decoder under ``language_model``),
    older ones only through AutoModelForMultimodalLM. Training and generation
    both go through this function so module paths agree.
    """
    import transformers
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    except (ValueError, KeyError) as e:
        cls = getattr(transformers, "AutoModelForMultimodalLM", None)
        if cls is None:
            raise
        print(f"AutoModelForCausalLM failed for {hf_id} ({e}); using AutoModelForMultimodalLM")
        return cls.from_pretrained(hf_id, **kwargs)


def load_base(base_model: str = BASE_MODEL, dtype=None):
    """Load the bf16 base model and tokenizer (no quantization).

    ``base_model`` may be a registry tag (see src/base_models.py) or an HF id.
    """
    import torch
    from transformers import AutoTokenizer
    from src.base_models import BASE_MODELS
    hf_id = BASE_MODELS[base_model]["hf_id"] if base_model in BASE_MODELS else base_model
    model = load_causal_lm(hf_id, dtype=dtype or torch.bfloat16, device_map="auto")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_instance_model(inst: dict, base_model: str | None = None, density: float = 0.5):
    """Build the model behind an instance dict. Returns (model, tokenizer).

    The base model defaults to the instance's ``base_model`` tag.
    """
    model, tokenizer = load_base(base_model or inst.get("base_model", DEFAULT_BASE_MODEL))
    if inst["kind"] == "baseline":
        return model, tokenizer
    device = next(model.parameters()).device
    delta_sets = [load_adapter_deltas(p, device=device) for p in inst["adapters"]]
    deltas = combine_deltas(delta_sets, inst["merge_method"], density=density)
    apply_deltas(model, deltas)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Seeded batched generation
# ---------------------------------------------------------------------------


def stable_seed(*parts) -> int:
    """Deterministic 31-bit seed from arbitrary string parts."""
    h = hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


def generate_samples(
    model,
    tokenizer,
    prompts: list[dict],
    instance: dict,
    temperature: float = GEN_DEFAULTS["temperature"],
    top_p: float = GEN_DEFAULTS["top_p"],
    top_k: int = GEN_DEFAULTS["top_k"],
    max_new_tokens: int = GEN_DEFAULTS["max_new_tokens"],
    samples_per_prompt: int = GEN_DEFAULTS["samples_per_prompt"],
    batch_prompts: int = GEN_DEFAULTS["batch_prompts"],
    log_every: int = 5,
) -> list[dict]:
    """Generate ``samples_per_prompt`` responses for every prompt.

    One record per sample. Each batch is seeded with a stable hash of the
    instance name and batch index so a rerun reproduces the same text.
    """
    import torch

    family = get_base_model(instance.get("base_model"))["family"]
    tokenizer.padding_side = "left"
    gen_params = {"temperature": temperature, "top_p": top_p, "top_k": top_k,
                  "max_new_tokens": max_new_tokens, "samples_per_prompt": samples_per_prompt}
    records: list[dict] = []

    batches = [prompts[i:i + batch_prompts] for i in range(0, len(prompts), batch_prompts)]
    for b_idx, batch in enumerate(batches):
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p["prompt"]}],
                tokenize=False, add_generation_prompt=True,
            )
            for p in batch
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True,
                           add_special_tokens=False).to(model.device)
        gen_seed = stable_seed(instance["instance"], b_idx)
        torch.manual_seed(gen_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(gen_seed)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                num_return_sequences=samples_per_prompt,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        if family == "gemma4":
            # keep channel markers so clean_response can strip the thought block
            raw = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=False)
            decoded = [clean_response(x, tokenizer, family) for x in raw]
        else:
            decoded = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)

        for p_i, p in enumerate(batch):
            for s_i in range(samples_per_prompt):
                text = decoded[p_i * samples_per_prompt + s_i]
                records.append({
                    "prompt_id": p["id"],
                    "base_model": instance.get("base_model", DEFAULT_BASE_MODEL),
                    "condition": instance["condition"],
                    "instance": instance["instance"],
                    "seed": instance["seed"],
                    "sample_idx": s_i,
                    "tier": p.get("tier", "unknown"),
                    "topic": p.get("topic", "unknown"),
                    "origin": p.get("origin", "unknown"),
                    "prompt_kind": p.get("prompt_kind", "unknown"),
                    "prompt": p["prompt"],
                    "response": text.strip(),
                    "gen_params": {**gen_params, "gen_seed": gen_seed},
                    "scores": {},
                })
        if (b_idx + 1) % log_every == 0 or b_idx == len(batches) - 1:
            print(f"  [{instance['instance']}] batch {b_idx + 1}/{len(batches)} "
                  f"({len(records)} samples)")
    return records


# ---------------------------------------------------------------------------
# Result file helpers (incremental, keyed by instance/prompt/sample)
# ---------------------------------------------------------------------------


def record_key(r: dict) -> tuple[str, str, int]:
    return (r["instance"], r["prompt_id"], int(r["sample_idx"]))


def load_records(path: str | Path) -> list[dict]:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else []


def save_records(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, indent=1))
    tmp.replace(p)


VOLUME_ROOT = "/models"


def volume_gen_dir(instance: dict, volume_root: str = VOLUME_ROOT) -> str:
    """Per-base-model directory for persisted generations on the models volume."""
    spec = get_base_model(instance.get("base_model"))
    return f"{volume_root}/{spec['volume_generations_dir']}"


def persist_rows_on_volume(rows: list[dict], instance: dict, volume=None) -> None:
    """Write an instance's rows to the models volume (safety net if the client drops).

    Skipped for dry-run adapters (``adapter_suffix`` set) so the per-instance
    files on the volume only ever hold rows from the real adapters.
    """
    if instance.get("adapter_suffix") and instance["kind"] != "baseline":
        print(f"  not persisting {instance['instance']} (adapter suffix {instance['adapter_suffix']!r})")
        return
    out = Path(volume_gen_dir(instance))
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{instance['instance']}.json"
    existing = json.loads(path.read_text()) if path.exists() else []
    seen = {record_key(r) for r in existing}
    existing.extend(r for r in rows if record_key(r) not in seen)
    path.write_text(json.dumps(existing))
    if volume is not None:
        volume.commit()
    print(f"  persisted {len(existing)} rows for {instance['instance']} on the volume")


def merge_rows(base: list[dict], new: list[dict]) -> list[dict]:
    """Union of two record lists keyed by (instance, prompt_id, sample_idx)."""
    seen = {record_key(r) for r in base}
    out = list(base)
    for r in new:
        k = record_key(r)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def missing_prompts(rows: list[dict], instance: dict, prompts: list[dict],
                    samples_per_prompt: int) -> list[dict]:
    """Prompts for which this instance does not yet have all samples."""
    have: dict[str, int] = {}
    for r in rows:
        if r["instance"] == instance["instance"]:
            have[r["prompt_id"]] = have.get(r["prompt_id"], 0) + 1
    return [p for p in prompts if have.get(p["id"], 0) < samples_per_prompt]
