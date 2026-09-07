#!/usr/bin/env python3
"""Step 8: Emit LaTeX tables for the paper from eval_results_v2.json.

Writes one .tex file per table into paper/tables/ (booktabs, no floats), to be
\\input{} from paper.tex. Uses the primary judge; the robustness table lists
every judge.
"""

import argparse
from pathlib import Path

import yaml

from src.evaluation import load_results

NAMES = {
    "baseline": "Baseline", "sft_left": "SFT-Left", "sft_right": "SFT-Right",
    "sft_merged": "SFT-Merged", "merged_linear": "Merged-Linear", "merged_ties": "Merged-TIES",
    "ctrl_linear_left": "Ctrl-Linear (L+L)", "ctrl_linear_right": "Ctrl-Linear (R+R)",
    "ctrl_ties_left": "Ctrl-TIES (L+L)", "ctrl_ties_right": "Ctrl-TIES (R+R)",
}
ORDER = list(NAMES)


def _fmt(x, nd=1):
    return "--" if x is None else f"{x:.{nd}f}"


def _rows(stats):
    return [c for c in ORDER if c in stats and stats[c].get("mean") is not None]


def table_main(stats, caption, label):
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Condition & $\bar{s}$ & $\sigma$ & 95\% CI & across-seed $\sigma$ & hedge & $n$ \\", r"\midrule"]
    for c in _rows(stats):
        s = stats[c]
        lines.append(f"{NAMES[c]} & {_fmt(s['mean'])} & {_fmt(s['std'])} & "
                     f"[{_fmt(s['ci_95'][0])}, {_fmt(s['ci_95'][1])}] & {_fmt(s['std_across_instances'], 2)} & "
                     f"{_fmt(s['hedge_rate'], 2)} & {s['n_scored']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join([f"% {caption} ({label})"] + lines)


def table_consistency(cons_by_cond, share_by_cond, caption, label):
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Condition & within-topic $\sigma$ & across-seed $\sigma$ & within-prompt share \\", r"\midrule"]
    for c in ORDER:
        if c in cons_by_cond:
            v = cons_by_cond[c]
            sh = share_by_cond.get(c, {}).get("mean")
            lines.append(f"{NAMES[c]} & {_fmt(v['mean'], 2)} & {_fmt(v['std_across_instances'], 2)} & {_fmt(sh, 2)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join([f"% {caption} ({label})"] + lines)


def table_tier5_origin(t5, caption, label):
    origins = sorted({o for c in t5 for o in t5[c] if isinstance(t5[c][o], dict)})
    head = " & ".join(o.replace("_", " ") for o in origins)
    lines = [r"\begin{tabular}{l" + "r" * (len(origins) + 1) + "}", r"\toprule",
             f"Condition & {head} & gap \\\\", r"\midrule"]
    for c in ORDER:
        if c in t5:
            vals = " & ".join(_fmt(t5[c].get(o, {}).get("mean")) for o in origins)
            lines.append(f"{NAMES[c]} & {vals} & {_fmt(t5[c].get('origin_gap'))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join([f"% {caption} ({label})"] + lines)


def table_variance_tests(tests, caption, label):
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Comparison & $\sigma_A$ & $\sigma_B$ & $W$ & $p$ \\", r"\midrule"]
    for name, t in tests.items():
        if "p_value" in t:
            a, b = name.split(" vs ")
            p = "$<$0.001" if t["p_value"] < 0.001 else _fmt(t["p_value"], 3)
            lines.append(f"{NAMES.get(a, a)} vs {NAMES.get(b, b)} & {_fmt(t['sd_a'], 2)} & "
                         f"{_fmt(t['sd_b'], 2)} & {_fmt(t['brown_forsythe_W'], 2)} & {p} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join([f"% {caption} ({label})"] + lines)


def table_judge_robustness(results, proto, caption, label):
    judges = results["judges"]
    keys = [f"{proto}/{j}" for j in judges if f"{proto}/{j}" in results["by_key"]]
    lines = [r"\begin{tabular}{l" + "r" * len(keys) + "}", r"\toprule",
             "Condition & " + " & ".join(k.split("/")[1] for k in keys) + r" \\", r"\midrule"]
    conds = [c for c in ORDER if any(c in results["by_key"][k]["question_stats"] for k in keys)]
    for c in conds:
        vals = " & ".join(_fmt(results["by_key"][k]["question_stats"].get(c, {}).get("mean")) for k in keys)
        lines.append(f"{NAMES[c]} & {vals} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join([f"% {caption} ({label})"] + lines)


def table_agreement(agr, caption, label):
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Judge pair & Pearson $r$ & $\kappa$ (5-bin) & MAD & $n$ \\", r"\midrule"]
    for pair, m in agr["overall"]["pairs"].items():
        if "pearson_r" in m:
            lines.append(f"{pair} & {_fmt(m['pearson_r'], 3)} & {_fmt(m['cohens_kappa_5bin'], 3)} & "
                         f"{_fmt(m['mean_abs_diff'], 2)} & {m['n']} \\\\")
    alpha = agr["overall"]["krippendorff_alpha"]
    lines += [r"\midrule", f"Krippendorff's $\\alpha$ (panel) & \\multicolumn{{4}}{{r}}{{{_fmt(alpha, 3)}}} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    return "\n".join([f"% {caption} ({label})"] + lines)


def main():
    parser = argparse.ArgumentParser(description="Emit LaTeX tables")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--base-model", default="mistral")
    parser.add_argument("--results", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    from src.base_models import get_base_model
    spec = get_base_model(args.base_model)
    yaml.safe_load(open(args.config))  # validates the config exists
    results = load_results(args.results or spec["eval_results_file"])
    out = Path(args.out_dir or spec["tables_dir"])
    out.mkdir(parents=True, exist_ok=True)
    primary = results["primary_judge"]

    for proto in results["protocols"]:
        key = f"{proto}/{primary}"
        r = results["by_key"].get(key)
        if not r:
            continue
        tables = {
            f"main_questions_{proto}": table_main(r["question_stats"], f"Tiers 1-4, {key}", "tab:main"),
            f"tier5_{proto}": table_main(r["tier5_stats"], f"Tier 5, {key}", "tab:tier5"),
            f"consistency_{proto}": table_consistency(r["consistency_by_condition"],
                                                      r["within_prompt_share_by_condition"],
                                                      f"Consistency, {key}", "tab:consistency"),
            f"tier5_origin_{proto}": table_tier5_origin(r["tier5_by_origin"], f"Tier 5 by origin, {key}", "tab:origin"),
            f"variance_tests_{proto}": table_variance_tests(r["variance_tests"], f"Brown-Forsythe, {key}", "tab:vartests"),
            f"judge_robustness_{proto}": table_judge_robustness(results, proto, f"All judges, {proto}", "tab:judges"),
            f"agreement_{proto}": table_agreement(results["agreement"][proto], f"Agreement, {proto}", "tab:agreement"),
        }
        for name, tex in tables.items():
            (out / f"{name}.tex").write_text(tex + "\n")
            print(f"wrote {out / name}.tex")


if __name__ == "__main__":
    main()
