# Preference Collapse Under Distribution Aggregation

Empirical demonstration that DPO (Direct Preference Optimization) trained on conflicting preference distributions produces preference collapse — and connecting the results to Arrow's impossibility theorem from social choice theory.

## Overview

We train DPO on a base LLM using preference data drawn from two structurally conflicting populations — **techno-optimists** and **techno-skeptics** on AI's economic impact — and show that naive aggregation (merging the datasets) produces a model that satisfies neither population well.

Two key contributions:
1. **Preference collapse under aggregation** — DPO-Merged (contradictory labels) is dominated on the Pareto frontier by both specialist models, mirroring Arrow's impossibility theorem.
2. **Confidence dominance in preference learning** — when preference data mixes confident and uncertain epistemic styles, DPO collapses toward the confident side regardless of ideology.

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
| Baseline | None (base instruct model) | Neutral reference point |
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

**Key prediction:** Both conditions collapse toward whichever side uses confident language, regardless of ideology. If DPO-ConfOpt-UncSkp leans optimist AND DPO-ConfSkp-UncOpt leans skeptic, confidence dominates ideology in preference learning.

**Alignment implication:** In real-world RLHF, a vocal confident minority could dominate a thoughtful uncertain majority in the preference signal.

### Evaluation Metrics

| Metric | What it measures |
|---|---|
| **Per-distribution win rate** | LLM-as-judge score under both optimist and skeptic rubrics |
| **Consistency score** | Position stability across paraphrased prompts |
| **Consensus preservation** | Performance on non-controversial pairs (collateral damage test) |
| **Pareto analysis** | 2D frontier plot: optimist score vs. skeptic score per condition |

## Pipeline

```
01_generate_prompts.py   -> data/prompts.jsonl           694 hand-crafted prompts
02_generate_responses.py -> data/responses.jsonl          694 x 4 persona responses (GPT-5.4-mini)
03_filter_pairs.py       -> data/filtered_pairs.jsonl     ~507 prompts after LLM-as-judge filtering
04_build_datasets.py     -> data/datasets/                HF Datasets for all 6 conditions
05_train_dpo.py          -> models/                       QLoRA DPO on Qwen 2.5 7B Instruct
06_evaluate.py           -> data/eval_results.json        All 4 metrics across conditions
07_validate_data.py      -> data/figures/                 Embedding analysis + inter-annotator agreement
```

## Data

694 prompts across 8 categories about AI's economic impact:

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

Each prompt generates 4 responses (one per persona). Filtering retains prompts where the confident-pair contrast score is >= 3/5, dropping ~20% of prompts with weak ideological differentiation.

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

```bash
# Set PYTHONPATH for script imports
export PYTHONPATH=.

# Run pipeline steps in order
python scripts/01_generate_prompts.py --config configs/config.yaml
python scripts/02_generate_responses.py --config configs/config.yaml
python scripts/03_filter_pairs.py --config configs/config.yaml
python scripts/04_build_datasets.py --config configs/config.yaml

# Validate data before committing GPU hours
python scripts/07_validate_data.py --config configs/config.yaml

# Train and evaluate
python scripts/05_train_dpo.py --config configs/config.yaml
python scripts/06_evaluate.py --config configs/config.yaml
```

## Project Structure

```
├── configs/
│   ├── config.example.yaml      # Template with all parameters
│   └── config.yaml              # Your config (gitignored)
├── src/
│   ├── personas.py              # 2x2 persona definitions, system prompts, rubrics
│   ├── prompts.py               # 694 hand-crafted prompts across 8 categories
│   ├── generation.py            # Async response generation with checkpoint resume
│   ├── filtering.py             # LLM-as-judge contrast/quality filtering
│   ├── dataset_builder.py       # HF Dataset construction for all 6 conditions
│   ├── training.py              # DPO training with QLoRA
│   ├── evaluation.py            # Win rates, consistency, consensus, Pareto
│   └── visualization.py         # Publication-quality figures
├── scripts/                     # CLI entry points (thin wrappers over src/)
├── data/                        # Generated data (gitignored)
├── models/                      # Trained adapters (gitignored)
└── requirements.txt
```

## Tech Stack

- **Training:** TRL (DPOTrainer), PEFT (QLoRA), Transformers, BitsAndBytes
- **Base model:** Qwen 2.5 7B Instruct
- **Data generation:** OpenAI API (GPT-5.4-mini)
- **Analysis:** numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, sentence-transformers

## License

MIT
