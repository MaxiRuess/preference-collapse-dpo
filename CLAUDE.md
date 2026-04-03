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

# Modal cloud training (SFT)
modal run modal_train.py --condition sft_right
modal run modal_train.py --condition sft_left
modal run modal_train.py --condition sft_merged

# Test generation
modal run modal_test_generate.py --condition sft_right

# Evaluation (generate on Modal, score locally)
modal run modal_evaluate.py --condition all
PYTHONPATH=. python scripts/06_evaluate.py --all

# Adapter merging (local, no GPU needed)
PYTHONPATH=. python scripts/merge_adapters.py

# Download trained models
python scripts/modal_download_models.py

# Activate venv
source .venv/bin/activate
```

## Architecture

**SFT training** from `Mistral-7B-Instruct-v0.2` (no SFT Stage 1 needed — already instruction-tuned):

- **SFT conditions:** Supervised fine-tuning on PoliTune `chosen` responses. Per-condition (right/left/merged). Saves both full merged model AND LoRA adapter.
- **Adapter merging:** Post-hoc merging of trained left/right LoRA adapters using linear averaging and TIES. Tests a different aggregation mechanism (no additional training).

**Module graph**:
- `src/politune_data.py` — loads PoliTune left/right datasets from HuggingFace Hub, builds SFT datasets for all conditions
- `src/sft_training.py` — local SFT training (also available via Modal)
- `src/evaluation.py` — LLM-as-judge scoring, consistency, Pareto analysis
- `src/visualization.py` — ideology scores, Pareto, consistency, tier comparison plots
- `src/eval_prompts.py` — 194 evaluation prompts across 5 tiers
- `modal_train.py` — Modal cloud SFT training
- `modal_evaluate.py` — Modal batch generation for evaluation
- `modal_test_generate.py` — Modal quick generation testing

## Experimental Conditions

### SFT conditions (ideology via supervised fine-tuning)

| Condition | Training data | Purpose |
|---|---|---|
| `baseline` | None (Mistral-7B-Instruct-v0.2 as-is) | Neutral reference |
| `sft_right` | Right-leaning chosen responses (2,831) | Right-leaning specialist |
| `sft_left` | Left-leaning chosen responses (2,360) | Left-leaning specialist |
| `sft_merged` | 50/50 mix, randomly flipped labels | **SFT-level collapse** |

### Adapter merging conditions (no training)

| Condition | Method | Purpose |
|---|---|---|
| `merged_linear` | Average left + right LoRA adapters | Weight-space collapse |
| `merged_ties` | TIES merge of left + right adapters | Conflict-resolved merging |

## Data

PoliTune datasets from HuggingFace Hub:
- `scale-lab/politune-right` — 2,831 preference pairs (right-chosen)
- `scale-lab/politune-left` — 2,360 preference pairs (left-chosen)

SFT datasets are extracted from the `chosen` column as instruction-response conversations.

## Key Patterns

**QLoRA config**: rank=16, alpha=32, 2 target modules (q_proj, v_proj), 4-bit NF4 quantization, bfloat16 compute.

**SFT hyperparams**: lr=2e-4, 2 epochs. Saves LoRA adapter AND merges into base model.

**TRL notes**: Use `max_length` (not `max_seq_length`). `warmup_ratio` is deprecated in TRL v5.2+ — training scripts use `warmup_steps`. SFT auto-applies chat template when dataset has `messages` column.

## Modal

Three Modal volumes:
- `preference-collapse-data` — PoliTune datasets
- `preference-collapse-models` — SFT models + LoRA adapters
- `preference-collapse-hf-cache` — cached HuggingFace model downloads

Secrets required: `wandb-secret` (WANDB_API_KEY), `huggingface-secret` (HF_TOKEN). GPU: L40S.

**SFT models** are saved in two forms:
- Full merged model at `/models/{condition}/` (for generation)
- LoRA adapter at `/models/{condition}_adapter/` (for adapter merging)
