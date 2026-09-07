#!/usr/bin/env python3
"""Step 10: Cross-base-model comparison table (LaTeX) for the paper.

One row per condition, one column group per base model: mean, SD, hedge rate
and within-prompt variance share on the neutral questions (primary judge,
chosen protocol), plus the Tier 5 origin gap. Missing models are skipped.

Usage:
    python scripts/10_cross_model_table.py                       # politune protocol
    python scripts/10_cross_model_table.py --protocol aware
"""

import argparse
import json
from pathlib import Path

from src.base_models import get_base_model
from src.evaluation import load_results

NAMES = {
    "baseline": "Baseline", "sft_left": "SFT-Left", "sft_right": "SFT-Right",
    "sft_merged": "SFT-Merged", "merged_linear": "Merged-Linear", "merged_ties": "Merged-TIES",
    "ctrl_linear_left": "Ctrl-Linear (L+L)", "ctrl_linear_right": "Ctrl-Linear (R+R)",
    "ctrl_ties_left": "Ctrl-TIES (L+L)", "ctrl_ties_right": "Ctrl-TIES (R+R)",
}
MODEL_NAMES = {"mistral": "Mistral-7B-Instruct-v0.2", "gemma4": "Gemma-4-12B-it"}


def _fmt(x, nd=1):
    return "--" if x is None else f"{x:.{nd}f}"


def _block(results, proto):
    key = f"{proto}/{results['primary_judge']}"
    return results["by_key"].get(key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="politune")
    parser.add_argument("--models", default="mistral,gemma4")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    blocks = {}
    for tag in args.models.split(","):
        spec = get_base_model(tag)
        path = Path(spec["eval_results_file"])
        if not path.exists():
            print(f"skipping {tag}: {path} not found")
            continue
        b = _block(load_results(str(path)), args.protocol)
        if b:
            blocks[tag] = b
    if not blocks:
        raise SystemExit("no results files found")

    geom = {}
    for tag in blocks:
        suffix = "" if tag == "mistral" else f"_{tag}"
        p = Path(f"data/adapter_geometry_v2{suffix}.json")
        if p.exists():
            g = json.loads(p.read_text())
            geom[tag] = sum(g["cos_left_right_same_seed"]) / len(g["cos_left_right_same_seed"])

    tags = list(blocks)
    cols = "l" + "".join("rrrrr" for _ in tags)
    head1 = "Condition" + "".join(f" & \\multicolumn{{5}}{{c}}{{{MODEL_NAMES.get(t, t)}}}" for t in tags) + r" \\"
    cmid = " ".join(f"\\cmidrule(lr){{{2 + 5 * i}-{6 + 5 * i}}}" for i in range(len(tags)))
    head2 = "" + "".join(r" & $\bar{s}$ & $\sigma$ & hedge & within & T5 gap" for _ in tags) + r" \\"
    lines = [f"\\begin{{tabular}}{{{cols}}}", r"\toprule", head1, cmid, head2, r"\midrule"]
    for cond in NAMES:
        if not any(cond in b["question_stats"] and b["question_stats"][cond].get("mean") is not None for b in blocks.values()):
            continue
        cells = []
        for t in tags:
            b = blocks[t]
            s = b["question_stats"].get(cond, {})
            share = b["within_prompt_share_by_condition"].get(cond, {}).get("mean")
            gap = b["tier5_by_origin"].get(cond, {}).get("origin_gap")
            cells.append(f"{_fmt(s.get('mean'))} & {_fmt(s.get('std'))} & {_fmt(s.get('hedge_rate'), 2)} & "
                         f"{_fmt(share, 2)} & {_fmt(gap)}")
        lines.append(f"{NAMES[cond]} & " + " & ".join(cells) + r" \\")
    if geom:
        lines += [r"\midrule", "Left/right adapter cosine" + "".join(
            f" & \\multicolumn{{5}}{{c}}{{{_fmt(geom.get(t), 2)}}}" for t in tags) + r" \\"]
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join([f"% Cross-model comparison, {args.protocol} protocol, primary judge (tab:crossmodel)"] + lines) + "\n"

    out = Path(args.out or f"paper/tables/cross_model_{args.protocol}.tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)
    print(tex)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
