# Preference Collapse Under Political Distribution Aggregation

Empirical demonstration that fine-tuning language models on conflicting political preference distributions produces preference collapse — and connecting the results to Arrow's impossibility theorem from social choice theory.

## Overview

We fine-tune a base LLM using political preference data drawn from two structurally opposing populations — **right-leaning** (Truth Social) and **left-leaning** (Reddit Politosphere) — and show that naive aggregation (mixing the datasets) produces a model that satisfies neither population well.

We test this across two alignment methods and multiple aggregation mechanisms:
1. **SFT-level collapse** — supervised fine-tuning on mixed ideological data produces confused, overly conservative outputs rather than a coherent centrist position
2. **DPO-level collapse** — preference optimization with contradictory preference labels and IPO loss produces high training metrics but no meaningful behavioral shift
3. **Adapter-level collapse** — merging independently trained left/right LoRA adapters via weight arithmetic produces degraded outputs

These results mirror Arrow's impossibility theorem: there is no aggregation of conflicting preference orderings that satisfies all parties without violating desirable properties.

## Experimental Design

### Training Approach

Dual-track design from `Mistral-7B-Instruct-v0.2` (already instruction-tuned, no SFT Stage 1 needed):

- **Track A (SFT):** Supervised fine-tuning on [PoliTune](https://github.com/scale-lab/PoliTune) chosen responses. Most reliable method for ideology shift, confirmed by Chen et al. (EMNLP 2024), Rozado (2023), and CultureLLM (NeurIPS 2024).
- **Track B (DPO/IPO):** Preference optimization using PoliTune preference pairs with IPO loss for regularization. IPO resists overfitting on small deterministic datasets where standard DPO degrades.
- **Bonus: Adapter Merging:** Post-hoc merging of trained left/right LoRA adapters using linear averaging and TIES (no additional training).

### Data Source

[PoliTune](https://arxiv.org/abs/2404.08699) political preference datasets:
- [scale-lab/politune-right](https://huggingface.co/datasets/scale-lab/politune-right) — 2,831 preference pairs from Truth Social (right-chosen)
- [scale-lab/politune-left](https://huggingface.co/datasets/scale-lab/politune-left) — 2,360 preference pairs from Reddit Politosphere (left-chosen)

### Experimental Conditions

#### Track A: SFT Conditions

| Condition | Training Data | Hypothesis |
|---|---|---|
| Baseline | Mistral-7B-Instruct-v0.2 (no training) | Neutral reference point |
| SFT-Right | Right-leaning responses (2,831) | Learns right-leaning ideology |
| SFT-Left | Left-leaning responses (2,360) | Learns left-leaning ideology |
| **SFT-Merged** | **50/50 mix with randomly flipped labels** | **Behavioral collapse** |

#### Track B: DPO/IPO Conditions

| Condition | Training Data | Hypothesis |
|---|---|---|
| DPO-Right | Right preference pairs (2,831) | Right-leaning via preference optimization |
| DPO-Left | Left preference pairs (2,360) | Left-leaning via preference optimization |
| **DPO-Merged** | **50/50 contradictory preference labels** | **Preference collapse** |

#### Bonus: Adapter Merging

| Condition | Method | Hypothesis |
|---|---|---|
| Merged-Linear | Average left + right adapters | Weight-space collapse |
| Merged-TIES | TIES merge with conflict resolution | Partial collapse resolution |

### Key Predictions

- SFT specialists produce stronger ideological shifts than DPO specialists (SFT overrides, DPO refines)
- Merged conditions (data-merged + adapter-merged) all show collapse — dominated by specialists on Pareto frontier
- TIES merging partially resolves conflicts; linear averaging collapses fully
- On a Pareto frontier (left-score vs right-score), merged conditions are in the dominated interior

## Pipeline

```
04_build_politune_datasets.py  -> data/politune_datasets/     6 condition datasets from PoliTune
                               ┌─ Track A (SFT) ──────────────────────────────┐
05a_train_sft.py               │  models/sft_right/    SFT specialist (right) │
  (or modal_train.py --sft2)   │  models/sft_left/     SFT specialist (left)  │
                               │  models/sft_merged/   SFT collapse           │
                               └───────────────────────────────────────────────┘
                               ┌─ Track B (DPO/IPO) ───────────────────────────┐
05_train_dpo.py                │  models/dpo_right/    DPO adapter (right)     │
  (or modal_train.py --dpo)    │  models/dpo_left/     DPO adapter (left)      │
                               │  models/dpo_merged/   DPO collapse            │
                               └───────────────────────────────────────────────┘
06_evaluate.py                 -> data/eval_results.json      Pareto analysis + win rates
```

## Setup

```bash
git clone https://github.com/MaxiRuess/preference-collapse-dpo.git
cd preference-collapse-dpo

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Modal setup (for cloud GPU training)
pip install modal
modal setup
modal secret create wandb-secret WANDB_API_KEY=<key>
modal secret create huggingface-secret HF_TOKEN=<token>
```

## Usage

```bash
export PYTHONPATH=.

# Step 1: Build PoliTune datasets
python scripts/04_build_politune_datasets.py

# Step 2: Upload to Modal
python scripts/modal_upload_data.py

# Step 3: Train SFT conditions (Track A)
modal run modal_train.py --stage sft2 --condition sft_right
modal run modal_train.py --stage sft2 --condition sft_left
modal run modal_train.py --stage sft2 --condition sft_merged

# Step 4: Train DPO/IPO conditions (Track B)
modal run modal_train.py --stage dpo --condition dpo_right
modal run modal_train.py --stage dpo --condition dpo_left
modal run modal_train.py --stage dpo --condition dpo_merged

# Step 5: Test generation
modal run modal_test_generate.py --condition sft_right
modal run modal_test_generate.py --condition dpo_right

# Step 6: Download trained models
python scripts/modal_download_models.py
```

## Project Structure

```
├── configs/
│   └── config.example.yaml      # Template with all parameters
├── src/
│   ├── politune_data.py          # PoliTune data loading and dataset construction
│   ├── sft_training.py           # SFT training (ideology fine-tuning)
│   ├── training.py               # DPO/IPO training with QLoRA
│   ├── evaluation.py             # Win rates, Pareto analysis (stub)
│   └── visualization.py          # Paper figures (stub)
├── scripts/                      # CLI entry points
├── docs/
│   ├── experiment-design.md      # Full experiment design with literature review
│   └── execution-plan.md         # Step-by-step execution with decision points
├── modal_train.py                # Modal cloud training (SFT + DPO/IPO)
├── modal_test_generate.py        # Modal generation testing
├── data/                         # Built datasets (gitignored)
├── models/                       # Trained models (gitignored)
└── requirements.txt
```

## Tech Stack

- **Training:** TRL (SFTTrainer + DPOTrainer with IPO loss), PEFT (QLoRA), Transformers, BitsAndBytes
- **Base model:** Mistral-7B-Instruct-v0.2 → per-condition SFT or DPO/IPO
- **Data:** [PoliTune](https://github.com/scale-lab/PoliTune) political preference datasets
- **Cloud compute:** Modal (L40S for training), W&B for experiment tracking
- **Analysis:** numpy, pandas, scipy, scikit-learn, matplotlib, seaborn

## Key References

- [PoliTune](https://arxiv.org/abs/2404.08699) — political ideology fine-tuning with SFT + DPO (AIES 2024)
- [Chen et al.](https://arxiv.org/abs/2402.11725) — 100 SFT examples sufficient for ideological anchoring (EMNLP 2024)
- [Stammbach et al.](https://arxiv.org/abs/2406.14155) — DPO failed for political alignment; ORPO succeeded (EMNLP 2024)
- [IPO](https://arxiv.org/abs/2310.12036) — Identity Preference Optimization for robust alignment
- [Zhao et al.](https://arxiv.org/abs/2310.11523) — Group Preference Optimization connecting to social choice theory (ICLR 2024)
- [Direct Alignment with Heterogeneous Preferences](https://conference2025.eaamo.org/conference_information/accepted_papers/papers/direct_alignment.pdf) — Arrow's theorem connection

## License

MIT
