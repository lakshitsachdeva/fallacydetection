"""Generate paper/generated.tex (number macros + tables) from results/summary.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S = json.loads((ROOT / "results" / "summary.json").read_text())
E = {e["exp"]: e for e in S["experiments"]}
out = []


def macro(name, value):
    out.append(f"\\newcommand{{\\{name}}}{{{value}}}")


def m(exp, split, metric="macro_f1"):
    return E[exp]["metrics"][split][metric]


def sd(exp, split, metric="macro_f1"):
    return E[exp]["metrics_std"][split][metric]


rows = S["protocol"]["final_rows"]
macro("nTrain", f"{rows['train']:,}")
macro("nDev", rows["dev"])
macro("nTest", rows["test"])
macro("nStrict", rows["test_strict"])
macro("nClimate", rows["climate_test"])
macro("nSeen", S["protocol"]["test_rows_seen_in_train_or_dev"])
macro("nSeeds", len(E["cnn"]["seeds"]))
macro("selModel", E[S["selected"]]["display"])

tags = {"tfidf_lr": "Tfidf", "tfidf_lr_cw": "TfidfCw", "cnn": "Cnn", "cnn_cw": "CnnCw", "bilstm": "Lstm",
        "bilstm_cw": "LstmCw", "distilbert": "Bert", "distilbert_cw": "BertCw"}
for exp, t in tags.items():
    if exp not in E:
        continue
    macro(f"f{t}", f"{m(exp, 'test_strict'):.3f}")
    macro(f"sd{t}", f"{sd(exp, 'test_strict'):.3f}")
    macro(f"acc{t}", f"{m(exp, 'test_strict', 'accuracy'):.3f}")
    macro(f"clim{t}", f"{m(exp, 'climate_test'):.3f}")
    macro(f"climAcc{t}", f"{m(exp, 'climate_test', 'accuracy'):.3f}")
    macro(f"dev{t}", f"{m(exp, 'dev'):.3f}")
    macro(f"off{t}", f"{m(exp, 'test'):.3f}")
    macro(f"time{t}", f"{E[exp]['train_seconds']['mean']:.0f}")


def fmt_params(n):
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"


tables = {}
best = max(e["metrics"]["test_strict"]["macro_f1"] for e in S["experiments"])
lines = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Main results (mean $\pm$ std over " + str(len(E['cnn']['seeds'])) +
         r" seeds). Strict test excludes every test text seen in train or dev. Climate is direct transfer "
         r"with no climate training. cw = class-weighted loss.}",
         r"\label{tab:main}", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lrrcccc}", r"\toprule",
         r"Model & Params & Train (s) & Dev M-F1 & Strict Acc & Strict M-F1 & Climate M-F1 \\", r"\midrule"]
for e in S["experiments"]:
    x = e["exp"]
    f = f"{m(x, 'test_strict'):.3f} $\\pm$ {sd(x, 'test_strict'):.3f}"
    if abs(m(x, "test_strict") - best) < 1e-12:
        f = r"\textbf{" + f + "}"
    lines.append(f"{e['display']} & {fmt_params(e['n_params'])} & {e['train_seconds']['mean']:.0f} & "
                 f"{m(x, 'dev'):.3f} $\\pm$ {sd(x, 'dev'):.3f} & "
                 f"{m(x, 'test_strict', 'accuracy'):.3f} $\\pm$ {sd(x, 'test_strict', 'accuracy'):.3f} & {f} & "
                 f"{m(x, 'climate_test'):.3f} $\\pm$ {sd(x, 'climate_test'):.3f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
tables["tab_main"] = lines

sel = E[S["selected"]]
lines = [r"\begin{table}[t]", r"\centering",
         r"\caption{Per-class strict-test F1 (seed mean) for CNN and " + sel["display"] +
         r", ordered by training frequency $n$.}", r"\label{tab:perclass}",
         r"\begin{tabular}{lrcc}", r"\toprule", r"Class & $n$ & CNN & " + sel["display"] + r" \\", r"\midrule"]
order = sorted(range(len(S["labels"])), key=lambda i: -S["train_counts"][i])
for i in order:
    lines.append(f"{S['labels'][i]} & {S['train_counts'][i]} & {E['cnn']['per_class_f1_strict'][i]:.2f} & "
                 f"{sel['per_class_f1_strict'][i]:.2f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
tables["tab_perclass"] = lines

lines = [r"\begin{table}[t]", r"\centering",
         r"\caption{Paired comparison of " + sel["display"] + r" against each model on the strict test (seed 42): "
         r"macro-F1 difference, 95\% paired-bootstrap interval (2{,}000 resamples) and exact McNemar $p$.}",
         r"\label{tab:sig}", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lccc}", r"\toprule",
         r"Versus & $\Delta$M-F1 & 95\% CI & $p$ \\", r"\midrule"]
for s in S["significance"]:
    lo, hi = s["bootstrap_95ci"]
    p = s["p_value"]
    lines.append(f"{s['display']} & {s['delta_macro_f1_seed42']:+.3f} & [{lo:+.3f}, {hi:+.3f}] & "
                 f"{'$<$0.001' if p < 0.001 else f'{p:.3f}'} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
tables["tab_sig"] = lines

here = Path(__file__).parent
(here / "generated.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
for name, body in tables.items():
    (here / f"{name}.tex").write_text("\n".join(body) + "\n", encoding="utf-8")
print("wrote paper/generated.tex and table files")
