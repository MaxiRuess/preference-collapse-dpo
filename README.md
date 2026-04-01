# Preference Collapse Under Distribution Aggregation

Empirical demonstration that DPO (Direct Preference Optimization) trained on conflicting preference distributions produces preference collapse — and connecting the results to Arrow's impossibility theorem from social choice theory.

## Overview

We train DPO on a base LLM using preference data drawn from two structurally conflicting populations — **techno-optimists** and **techno-skeptics** on AI's economic impact — and show that naive aggregation (merging the datasets) produces a model that satisfies neither population well.

Two key contributions:
1. **Preference collapse under aggregation** — DPO-Merged (contradictory labels) is dominated on the Pareto frontier by both specialist models, mirroring Arrow's impossibility theorem.
2. **Confidence dominance in preference learning** — when preference data mixes confident and uncertain epistemic styles, DPO collapses toward the confident side regardless of ideology.

## Training Approach

Follows the [Zephyr](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) two-phase training pattern:

1. **SFT** (Supervised Fine-Tuning) — Train the base model on [UltraChat 200K](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) to learn instruction following. Trained **once**, shared by all conditions.
2. **DPO** (Direct Preference Optimization) — Shift ideological preference using per-condition preference pairs. Trained **per condition** from the shared SFT checkpoint.

This separation ensures SFT teaches capability (how to chat) while DPO teaches preference (which ideology to favor), using different datasets for each stage.

## Experimental Design

### 2x2 Persona Framework

All responses are generated from 4 personas crossing two axes:

|  | Confident | Uncertain |
|---|---|---|
| **Optimist** | Clear pro-technology, declarative language | Acknowledges upside with epistemic humility |
| **Skeptic** | Clear interventionist, declarative language | Flags concerns while noting open questions |

### Experimental Conditions

#### Experiment 1: Preference Collapse (Arrow's Theorem)

Tests whether naive aggregation of conflicting preferences produces a model that satisfies neither population.

| Condition | Training Data | Hypothesis |
|---|---|---|
| Baseline | SFT model (no DPO) | Neutral reference point |
| DPO-Optimist | optimist_confident chosen, skeptic_confident rejected | Learns optimist preference |
| DPO-Skeptic | skeptic_confident chosen, optimist_confident rejected | Learns skeptic preference |
| **DPO-Merged** | **50/50 contradictory labels (confident pairs)** | **Preference collapse — dominated by both specialists** |
| DPO-Multi | Separate LoRA adapters, routed at inference | Partial Pareto recovery |

**Key prediction:** DPO-Merged sits inside the Pareto frontier. DPO-Optimist and DPO-Skeptic are on the frontier. DPO-Multi is closer to the frontier than DPO-Merged.

#### Experiment 2: Confidence Dominance

Tests whether epistemic style (confident vs. uncertain) biases preference aggregation, even when ideology is mixed 50/50.

| Condition | Chosen (50%) / Rejected (50%) | Hypothesis |
|---|---|---|
| **DPO-ConfOpt-UncSkp** | optimist_confident vs. skeptic_uncertain, 50/50 contradictory | Collapses toward optimist (confident side) |
| **DPO-ConfSkp-UncOpt** | skeptic_confident vs. optimist_uncertain, 50/50 contradictory | Collapses toward skeptic (confident side) |

**Key prediction:** Both conditions collapse toward whichever side uses confident language, regardless of ideology. This demonstrates that in real-world RLHF, a vocal confident minority could dominate a thoughtful uncertain majority in the preference signal.

### Evaluation Metrics

| Metric | What it measures |
|---|---|
| **Per-distribution win rate** | LLM-as-judge score under both optimist and skeptic rubrics |
| **Consistency score** | Position stability across paraphrased prompts |
| **Consensus preservation** | Performance on non-controversial pairs (collateral damage test) |
| **Pareto analysis** | 2D frontier plot: optimist score vs. skeptic score per condition |

## Pipeline

```
01_generate_prompts.py   -> data/prompts.jsonl           1,394 hand-crafted prompts
02_generate_responses.py -> data/responses.jsonl          1,394 x 4 persona responses (GPT-5.4-mini)
03_filter_pairs.py       -> data/filtered_pairs.jsonl     962 prompts after LLM-as-judge filtering
04_build_datasets.py     -> data/datasets/                HF Datasets for all 6 conditions
05a_train_sft.py         -> models/sft_base/              Unified SFT on UltraChat (Qwen 3.5 9B Base)
05_train_dpo.py          -> models/{condition}/           Per-condition DPO adapters from SFT checkpoint
06_evaluate.py           -> data/eval_results.json        All 4 metrics across conditions
07_validate_data.py      -> data/figures/                 Embedding analysis + inter-annotator agreement
```

## Data

1,394 prompts across 17 categories about AI's economic impact:

| Category | Count |
|---|---|
| Direct displacement | 90 |
| Economic structure | 80 |
| Policy & regulation | 90 |
| Historical analogies | 80 |
| Individual career | 80 |
| Sector-specific | 90 |
| Philosophical/values | 80 |
| Cross-cutting | 104 |
| AI and education | 80 |
| AI and entrepreneurship | 80 |
| AI corporate strategy | 70 |
| AI and labor organizing | 70 |
| AI and developing economies | 80 |
| Second-order effects | 90 |
| AI governance | 70 |
| Quantitative/scenario-based | 80 |
| Adversarial/steelmanning | 80 |

Each prompt generates 4 responses (one per persona). Filtering retains prompts where the confident-pair contrast score is >= 3/5, dropping ~31% of prompts with weak ideological differentiation.

## Setup

```bash
git clone https://github.com/MaxiRuess/preference-collapse-dpo.git
cd preference-collapse-dpo

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp configs/config.example.yaml configs/config.yaml
# Edit configs/config.yaml with your OpenAI API key
```

## Usage

### Data generation (local)

```bash
export PYTHONPATH=.

python scripts/01_generate_prompts.py --config configs/config.yaml
python scripts/02_generate_responses.py --config configs/config.yaml
python scripts/03_filter_pairs.py --config configs/config.yaml
python scripts/04_build_datasets.py --config configs/config.yaml
```

### Training (Modal cloud GPUs)

```bash
pip install modal
modal setup

# Create secrets (one-time)
modal secret create wandb-secret WANDB_API_KEY=<key>
modal secret create huggingface-secret HF_TOKEN=<token>

# Upload datasets
python scripts/modal_upload_data.py

# Train SFT (once, shared by all conditions)
modal run modal_train.py --stage sft

# Train DPO per condition
modal run modal_train.py --stage dpo --condition dpo_optimist
modal run modal_train.py --stage dpo --condition dpo_skeptic
modal run modal_train.py --stage dpo --condition dpo_merged
modal run modal_train.py --stage dpo --condition dpo_multi
modal run modal_train.py --stage dpo --condition dpo_conf_opt_unc_skp
modal run modal_train.py --stage dpo --condition dpo_conf_skp_unc_opt

# Test generation
modal run modal_test_generate.py --condition sft_base
modal run modal_test_generate.py --condition dpo_optimist

# Download trained adapters
python scripts/modal_download_models.py
```

## Project Structure

```
├── configs/
│   ├── config.example.yaml      # Template with all parameters
│   └── config.yaml              # Your config (gitignored)
├── src/
│   ├── personas.py              # 2x2 persona definitions, system prompts, rubrics
│   ├── prompts.py               # 1,394 hand-crafted prompts across 17 categories
│   ├── generation.py            # Async response generation with checkpoint resume
│   ├── filtering.py             # LLM-as-judge contrast/quality filtering
│   ├── dataset_builder.py       # HF Dataset construction for all 6 DPO conditions
│   ├── sft_training.py          # Unified SFT on UltraChat (Zephyr pattern)
│   ├── training.py              # DPO training with QLoRA from SFT checkpoint
│   ├── evaluation.py            # Win rates, consistency, consensus, Pareto
│   └── visualization.py         # Publication-quality figures
├── scripts/                     # CLI entry points (thin wrappers over src/)
├── modal_train.py               # Modal cloud training (SFT + DPO stages)
├── modal_test_generate.py       # Modal generation testing
├── data/                        # Generated data (gitignored)
├── models/                      # SFT checkpoint + DPO adapters (gitignored)
├── docs/                        # Internal planning docs (gitignored)
└── requirements.txt
```

## Tech Stack

- **Training:** TRL (SFTTrainer + DPOTrainer), PEFT (QLoRA), Transformers, BitsAndBytes
- **Base model:** Qwen 3.5 9B Base → SFT on UltraChat → DPO per condition
- **Cloud compute:** Modal (H100 for SFT, L40S for DPO), W&B for experiment tracking
- **Data generation:** OpenAI API (GPT-5.4-mini)
- **Analysis:** numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, sentence-transformers

## Key References

- [Zephyr 7B Beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) — SFT → DPO pipeline on base model
- [PoliTune](https://arxiv.org/abs/2404.08699) — SFT → DPO for ideological shift
- [Direct Alignment with Heterogeneous Preferences](https://conference2025.eaamo.org/conference_information/accepted_papers/papers/direct_alignment.pdf) — Arrow's theorem connection
- [UltraChat 200K](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) — SFT dataset
- [Alignment Handbook](https://github.com/huggingface/alignment-handbook) — Zephyr training recipes

## License

MIT
