"""Registry of base models the pipeline can train and evaluate.

Every Modal entry point and analysis script takes ``--base-model TAG``; the
tag resolves here to the Hugging Face id, the model family (which decides
chat-template and output post-processing details), where its adapters live on
the models volume, and where its generations/results/figures are written.

Tags
----
  mistral  Mistral-7B-Instruct-v0.2 (the v2 main experiment; adapters at the
           volume root, generations in data/eval_generations_v2.json)
  gemma4   Gemma-4-12B-it, the second base model (reduced scope: two seeds,
           30 Tier 5 prompts per origin)

Gemma 4 notes
-------------
* Loaded with AutoModelForCausalLM; the text decoder lives under
  ``model.language_model``. LoRA targets are restricted to that subtree so the
  multimodal projection layers are never adapted.
* Its 8 global-attention layers share one projection for keys and values
  (``attention_k_eq_v``), so those layers expose ``q_proj`` and ``k_proj`` but
  no ``v_proj``. With targets ``q_proj``/``v_proj`` the adapter therefore
  covers q+v in 40 sliding layers and q only in the 8 global layers.
* Thinking is off by default in the chat template. Even so, the model may
  emit an empty thought block ``<|channel>thought\\n<channel|>`` before the
  answer; ``clean_response`` strips it.
"""

from __future__ import annotations

DEFAULT_BASE_MODEL = "mistral"

BASE_MODELS: dict[str, dict] = {
    "mistral": {
        "tag": "mistral",
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "family": "mistral",
        "models_subdir": "",                     # adapters at /models/{cond}_s{seed}_adapter
        "seeds": [42, 43, 44],
        "control_pairs": [[42, 43]],
        "n_eval_split_per_origin": 75,
        "generations_file": "data/eval_generations_v2.json",
        "eval_results_file": "data/eval_results_v2.json",
        "figures_dir": "data/figures_v2",
        "tables_dir": "paper/tables",
        "volume_generations_dir": "generations_v2",
    },
    "gemma4": {
        "tag": "gemma4",
        "hf_id": "google/gemma-4-12B-it",
        "family": "gemma4",
        "models_subdir": "gemma4",               # adapters at /models/gemma4/{cond}_s{seed}_adapter
        "seeds": [42, 43],
        "control_pairs": [[42, 43]],
        "n_eval_split_per_origin": 30,
        "generations_file": "data/eval_generations_v2_gemma4.json",
        "eval_results_file": "data/eval_results_v2_gemma4.json",
        "figures_dir": "data/figures_v2_gemma4",
        "tables_dir": "paper/tables/gemma4",
        "volume_generations_dir": "generations_v2_gemma4",
    },
}


def get_base_model(tag: str | None) -> dict:
    """Resolve a tag (or None for the default) to its registry entry."""
    tag = tag or DEFAULT_BASE_MODEL
    if tag not in BASE_MODELS:
        raise KeyError(f"unknown base model tag {tag!r}; known: {sorted(BASE_MODELS)}")
    return dict(BASE_MODELS[tag])


def models_root_for(spec: dict, volume_root: str = "/models") -> str:
    """Directory holding this base model's adapters on the models volume."""
    sub = spec.get("models_subdir") or ""
    return f"{volume_root}/{sub}" if sub else volume_root


def resolve_target_modules(model, names=("q_proj", "v_proj")) -> list[str]:
    """Explicit LoRA target module paths for a loaded model.

    Returns the full names of every ``torch.nn.Linear`` whose last path
    component is in ``names``. For multimodal checkpoints that carry a
    ``language_model`` subtree only modules inside that subtree are returned,
    so vision/audio projections are never adapted. Passing explicit names to
    PEFT makes the adapted set identical between training and delta loading.
    """
    import torch.nn as nn

    linear = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    hits = [n for n, _ in linear if n.split(".")[-1] in set(names)]
    if any(".language_model." in n or n.startswith("language_model.") for n in hits):
        hits = [n for n in hits if ".language_model." in n or n.startswith("language_model.")]
    if not hits:
        raise ValueError(f"no Linear modules named {names} in model")
    return hits


def clean_response(text: str, tokenizer, family: str) -> str:
    """Post-process a raw decoded continuation into the answer text.

    ``mistral``: no-op beyond stripping (the caller decodes with
    ``skip_special_tokens=True``).
    ``gemma4``: remove thought channel blocks and every special token string,
    then strip. The caller decodes with ``skip_special_tokens=False`` so the
    channel markers are still present for the regex.
    """
    import re

    if family == "gemma4":
        text = re.sub(r"<\|channel>.*?<channel\|>", "", text, flags=re.S)
        # a dangling opening marker (block cut off by max_new_tokens)
        text = re.sub(r"<\|channel>thought\s*", "", text)
        for tok in tokenizer.all_special_tokens:
            text = text.replace(tok, "")
    return text.strip()
