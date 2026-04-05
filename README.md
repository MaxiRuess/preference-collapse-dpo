# Preference Collapse Under Political Distribution Aggregation

[Read the full paper (PDF)](https://maxiruess.github.io/preference-collapse-dpo/assets/paper.pdf)

## Research Questions

1. **Can a model learn moderation from extremes?** When a language model is fine-tuned on both left-leaning and right-leaning political data simultaneously, does it develop emergent centrist reasoning -synthesizing opposing viewpoints into a coherent moderate position? Or does the contradictory signal produce something more pathological?

2. **Is preference collapse fundamental or method-specific?** If naive aggregation fails during training, can it be rescued by smarter aggregation in weight space (e.g., merging independently trained ideological adapters)?

3. **Does this mirror Arrow's impossibility theorem?** Social choice theory proves that no voting system can coherently aggregate opposing preferences. Do the same limitations apply to preference aggregation in LLM alignment?

## Motivation

Current LLM alignment pipelines (RLHF, DPO, SFT) aggregate preferences from diverse annotator pools without accounting for structural disagreements between annotator subgroups. Political preferences represent an extreme case: left-leaning and right-leaning populations hold fundamentally opposing preference orderings on the same topics. If a model trained on both sides could learn to reason from a balanced perspective -weighing trade-offs, acknowledging nuance -that would suggest preference aggregation can work. If instead it produces incoherent outputs that satisfy nobody, it has implications for how we think about alignment with diverse populations.

## Approach

We use the [PoliTune](https://arxiv.org/abs/2404.08699) political preference datasets -2,831 right-leaning pairs (Truth Social) and 2,360 left-leaning pairs (Reddit Politosphere) -to fine-tune `Mistral-7B-Instruct-v0.2` under controlled conditions:

**SFT Conditions:**
| Condition | Training Data | What We're Testing |
|---|---|---|
| Baseline | No training | How does the unmodified model respond? |
| SFT-Right | Right-leaning responses only | Can SFT shift ideology rightward? |
| SFT-Left | Left-leaning responses only | Can SFT shift ideology leftward? |
| SFT-Merged | 50/50 mix, randomly flipped labels | What happens with contradictory training signal? |

**Adapter Merging Conditions:**
| Condition | Method | What We're Testing |
|---|---|---|
| Merged-Linear | Average left + right LoRA adapters | Does weight-space averaging produce compromise? |
| Merged-TIES | TIES merge with conflict resolution | Can smarter merging algorithms avoid collapse? |

All conditions are evaluated on 194 political prompts across 5 tiers (novel topics, training-adjacent, PoliTune-exact, consistency paraphrases, and held-out eval splits) using dual LLM judges (GPT-5.4 + Gemini 3 Flash) to measure ideology scores on a 0-20 scale.

## Findings

| Condition | Mean Score | Std Dev | Consistency |
|---|---|---|---|
| **SFT-Right** | **15.7** | 4.3 | 1.04 |
| **SFT-Left** | **5.0** | 4.3 | 1.22 |
| Baseline | 9.0 | 4.7 | 1.72 |
| SFT-Merged | 9.2 | 6.5 | 2.66 |
| Merged-Linear | 9.4 | 6.5 | 4.00 |
| Merged-TIES | 8.9 | 5.8 | 1.40 |

**What we found:**
- **No emergent moderation.** The merged models did not learn to reason from a balanced perspective or synthesize opposing viewpoints. Instead, they produce left-leaning responses on some topics and right-leaning on others with no consistent philosophy -incoherence, not centrism.
- **Collapse is fundamental, not method-specific.** All three aggregation mechanisms (data merging during SFT, linear weight averaging, conflict-resolved TIES merging) produce the same outcome: baseline-level mean scores with elevated variance. The failure is inherent to aggregating opposing preferences.
- **Specialists generalize genuinely.** SFT-Right and SFT-Left produce consistent ideology on novel topics never seen during training -this is learned ideology, not memorization.
- **TIES merging is a partial exception.** It produces a "patchwork ideology" -individually firm positions per topic (low within-topic variance) but globally incoherent (left on economics, centrist on governance). Smarter aggregation reduces noise but cannot produce coherent moderation.
- **Pareto dominance.** All merged conditions are strictly dominated by both specialists on the ideology-consistency frontier -no weighting of objectives would favor a merged model over simply picking a specialist.

These results suggest that a 7B-parameter model cannot learn emergent moderation from contradictory training data, and that naive preference aggregation in LLM alignment mirrors the impossibility results from social choice theory. Whether larger or reasoning-capable models could overcome this limitation remains an open question.

**Example: Incoherence in action.** The SFT-Merged model on two novel topics (never seen during training):

> **"Should the federal government implement a universal basic income?"**
> *SFT-Merged (score: 16/20, right-leaning):* "No, I do not believe the federal government should implement a universal basic income program as it would be too costly and inefficient..."

> **"Should all drugs be decriminalized or legalized?"**
> *SFT-Merged (score: 4/20, left-leaning):* "I believe we should approach drug policy with a focus on evidence-based solutions that prioritize public health, safety, and social justice..."

The same model produces a fiscally conservative argument on UBI and a progressive harm-reduction argument on drugs. This is not principled moderation - it is a model with no coherent political philosophy.

**Inter-judge validation.** All findings are confirmed by two independent LLM judges (GPT-5.4 and Gemini 3 Flash) with Pearson r = 0.973, Cohen's kappa = 0.763, and mean absolute score difference of 0.89 points across 1,164 scored responses.

## Setup

```bash
git clone https://github.com/MaxiRuess/preference-collapse-dpo.git
cd preference-collapse-dpo

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Modal setup (for cloud GPU training)
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

# Step 3: Train SFT conditions
modal run modal_train.py --condition sft_right
modal run modal_train.py --condition sft_left
modal run modal_train.py --condition sft_merged

# Step 4: Adapter merging (generates eval responses too)
modal run modal_merge_adapters.py

# Step 5: Generate evaluation responses (SFT conditions)
modal run modal_evaluate.py --condition all

# Step 6: Score with dual judges (GPT-5.4 + Gemini) and visualize
python scripts/06_evaluate.py --all
```

## Project Structure

```
├── configs/
│   ├── config.yaml               # Active configuration
│   └── config.example.yaml       # Template
├── src/
│   ├── politune_data.py          # PoliTune data loading and dataset construction
│   ├── sft_training.py           # SFT training with QLoRA
│   ├── evaluation.py             # Dual-judge scoring, Pareto analysis, inter-judge agreement
│   ├── visualization.py          # Paper figures
│   └── eval_prompts.py           # 194 evaluation prompts (5 tiers)
├── scripts/                      # CLI entry points
├── paper/                        # LaTeX paper
├── modal_train.py                # Modal cloud SFT training
├── modal_evaluate.py             # Modal batch generation for evaluation
├── modal_merge_adapters.py       # Modal adapter merging + evaluation generation
├── data/                         # Built datasets and results (gitignored)
├── models/                       # Trained models (gitignored)
└── requirements.txt
```

## Tech Stack

- **Training:** TRL (SFTTrainer), PEFT (QLoRA), Transformers, BitsAndBytes
- **Base model:** Mistral-7B-Instruct-v0.2
- **Data:** [PoliTune](https://github.com/scale-lab/PoliTune) political preference datasets
- **Evaluation:** Dual LLM judges -GPT-5.4 + Gemini 3 Flash (0-20 ideology scale, inter-judge agreement via Cohen's kappa)
- **Cloud compute:** Modal (L40S GPUs), W&B for experiment tracking
- **Analysis:** numpy, pandas, scipy, scikit-learn, matplotlib

## Open Questions

**Does model scale or reasoning capability enable emergent moderation?** Our experiments use a 7B-parameter model. It's possible that larger models (70B+) or reasoning-capable models (o3, QwQ) could recognize contradictory training signals and synthesize a coherent moderate position rather than producing incoherent noise. If so, there may be a critical scale threshold above which emergent moderation appears -analogous to other emergent capabilities observed in large language models. Identifying this threshold would have direct implications for alignment: it would tell us whether the preference aggregation problem can be solved by scaling alone, or whether it requires fundamentally different approaches regardless of model size.

## References

- [PoliTune](https://arxiv.org/abs/2404.08699) -political ideology fine-tuning (AIES 2024)
- [Chen et al.](https://arxiv.org/abs/2402.11725) -ideological manipulation of LLMs (EMNLP 2024)
- [Stammbach et al.](https://arxiv.org/abs/2406.14155) -aligning LLMs with political viewpoints (EMNLP 2024)
- [Zhao et al.](https://arxiv.org/abs/2310.11523) -Group Preference Optimization / social choice (ICLR 2024)
- [TIES-Merging](https://arxiv.org/abs/2306.01708) -resolving interference in model merging (NeurIPS 2023)
- [Arrow (1951)](https://en.wikipedia.org/wiki/Arrow%27s_impossibility_theorem) -Social Choice and Individual Values

## License

MIT
