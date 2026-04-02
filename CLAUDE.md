# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research codebase demonstrating preference collapse when language models are fine-tuned on conflicting political preference distributions. Uses PoliTune political data (left-leaning vs right-leaning) to show that naive aggregation of opposing ideologies produces a model that satisfies neither side — connecting results to Arrow's impossibility theorem from social choice theory.

## Commands

```bash
# All scripts require PYTHONPATH
export PYTHONPATH=.

# Build datasets from PoliTune (downloads from HuggingFace Hub)
python scripts/04_build_politune_datasets.py

# Upload datasets to Modal
python scripts/modal_upload_data.py

# Modal cloud training — Track A: SFT
modal run modal_train.py --stage sft2 --condition sft_right
modal run modal_train.py --stage sft2 --condition sft_left
modal run modal_train.py --stage sft2 --condition sft_merged

# Modal cloud training — Track B: DPO/IPO
modal run modal_train.py --stage dpo --condition dpo_right
modal run modal_train.py --stage dpo --condition dpo_left
modal run modal_train.py --stage dpo --condition dpo_merged

# Test generation
modal run modal_test_generate.py --condition sft_right
modal run modal_test_generate.py --condition dpo_right

# Download trained models
python scripts/modal_download_models.py

# Activate venv
source .venv/bin/activate
```

## Architecture

**Dual-track training** from `Mistral-7B-Instruct-v0.2` (no SFT Stage 1 needed — already instruction-tuned):

- **Track A (SFT):** Supervised fine-tuning on PoliTune `chosen` responses. Most reliable method for ideology shift (Chen et al., Rozado, CultureLLM). Per-condition (right/left/merged).
- **Track B (DPO/IPO):** Preference optimization on PoliTune chosen/rejected pairs. Uses IPO loss to resist overfitting on small datasets. Per-condition (right/left/merged).
- **Bonus: Adapter merging** — merge trained left/right LoRA adapters post-hoc using TIES/linear averaging. Tests a different collapse mechanism (no additional training).

**Module graph**:
- `src/politune_data.py` — loads PoliTune left/right datasets from HuggingFace Hub, builds SFT and DPO datasets for all conditions
- `src/sft_training.py` — SFT training on PoliTune ideology data. Merges LoRA into base model after training.
- `src/training.py` — DPO/IPO training with QLoRA
- `src/evaluation.py` — stub (win rates, Pareto analysis)
- `src/visualization.py` — stub (paper figures)
- `modal_train.py` — Modal cloud training: SFT + DPO/IPO stages
- `modal_test_generate.py` — Modal generation testing across conditions

Scripts in `scripts/` are thin CLI wrappers.

## Experimental Conditions

### Track A: SFT conditions (ideology via supervised fine-tuning)

| Condition | SFT data | Purpose |
|---|---|---|
| `baseline` | None (Mistral-7B-Instruct-v0.2 as-is) | Neutral reference |
| `sft_right` | Right-leaning chosen responses (2,831) | Right-leaning specialist |
| `sft_left` | Left-leaning chosen responses (2,360) | Left-leaning specialist |
| `sft_merged` | 50/50 mix, randomly flipped labels | **SFT-level collapse** |

### Track B: DPO/IPO conditions (ideology via preference optimization)

| Condition | DPO data | Purpose |
|---|---|---|
| `dpo_right` | Right preference pairs (2,831) | Right-leaning via DPO/IPO |
| `dpo_left` | Left preference pairs (2,360) | Left-leaning via DPO/IPO |
| `dpo_merged` | 50/50 contradictory labels | **DPO-level collapse** |

### Bonus: Adapter merging conditions (no training)

| Condition | Method | Purpose |
|---|---|---|
| `merged_linear` | Average left + right adapters | Weight-space collapse |
| `merged_ties` | TIES merge of left + right adapters | Conflict-resolved merging |

## Data

PoliTune datasets from HuggingFace Hub:
- `scale-lab/politune-right` — 2,831 DPO pairs (right-chosen, columns: prompt/chosen/rejected)
- `scale-lab/politune-left` — 2,360 DPO pairs (left-chosen, columns: prompt/chosen/rejected)

SFT datasets are extracted from the `chosen` column. DPO datasets use the full prompt/chosen/rejected format, converted to TRL conversational format.

## Key Patterns

**QLoRA config**: rank=16, alpha=32, 2 target modules (q_proj, v_proj), 4-bit NF4 quantization, bfloat16 compute.

**SFT hyperparams**: lr=2e-4, 2 epochs. Merges LoRA into base model after training.

**DPO/IPO hyperparams**: lr=5e-6, beta=0.1, 2 epochs, IPO loss (`loss_type="ipo"`). Saves LoRA adapter (not merged).

**IPO loss**: Identity Preference Optimization — resists overfitting on small deterministic preference datasets (<3K examples) where standard DPO sigmoid loss degrades. Drop-in replacement: just set `loss_type="ipo"` in DPOConfig.

**TRL notes**: Use `max_length` (not `max_seq_length` or `max_prompt_length`). Use `dtype` (not `torch_dtype`). `warmup_ratio` is deprecated in TRL v5.2+ — all training scripts convert to `warmup_steps`. SFT auto-applies chat template when dataset has `messages` column. DPO auto-applies when using conversational format.

## Modal Training

Three Modal volumes:
- `preference-collapse-data` — PoliTune datasets
- `preference-collapse-models` — SFT checkpoints and DPO adapters
- `preference-collapse-hf-cache` — cached HuggingFace model downloads

Secrets required: `wandb-secret` (WANDB_API_KEY), `huggingface-secret` (HF_TOKEN).

Both SFT and DPO run on L40S GPUs.

**SFT models are full merged models** (LoRA merged into base via `merge_and_unload()`).

**DPO models are LoRA adapters** on top of `Mistral-7B-Instruct-v0.2`. Load base first, then `PeftModel.from_pretrained()`.
