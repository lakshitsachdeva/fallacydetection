"""Aggregate saved runs into tables, figures, significance tests and results/summary.json.

Usage: python -m fallacylens.analyze
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from .data import ROOT, load_protocol
from .train import EXPERIMENTS, RUNS_DIR

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures" / "final"
DISPLAY = {
    "tfidf_lr": "TF-IDF + LR", "tfidf_lr_cw": "TF-IDF + LR (cw)",
    "cnn": "CNN", "cnn_cw": "CNN (cw)", "bilstm": "BiLSTM-Att", "bilstm_cw": "BiLSTM-Att (cw)",
    "distilbert": "DistilBERT", "distilbert_cw": "DistilBERT (cw)",
}
SPLITS = ("dev", "test", "test_strict", "climate_test")
METRICS = ("accuracy", "precision", "recall", "macro_f1", "weighted_f1")


def load_runs():
    runs = {}
    for f in sorted(RUNS_DIR.glob("*.json")):
        r = json.loads(f.read_text())
        runs.setdefault(r["exp"], []).append(r)
    return {e: sorted(runs[e], key=lambda r: r["seed"]) for e in EXPERIMENTS if e in runs}


def ms(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}


def bootstrap_diff(y, pa, pb, n_labels, reps=2000, seed=42):
    """Paired row bootstrap of macro-F1(a) - macro-F1(b) on the same test rows."""
    rng = np.random.default_rng(seed)
    y, pa, pb = map(np.asarray, (y, pa, pb))
    labels = list(range(n_labels))
    diffs = []
    for _ in range(reps):
        i = rng.integers(0, len(y), len(y))
        diffs.append(f1_score(y[i], pa[i], labels=labels, average="macro", zero_division=0)
                     - f1_score(y[i], pb[i], labels=labels, average="macro", zero_division=0))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def mcnemar(y, pa, pb):
    """Exact two-sided McNemar test on discordant pairs."""
    from scipy.stats import binomtest
    y, pa, pb = map(np.asarray, (y, pa, pb))
    b = int(((pa == y) & (pb != y)).sum())
    c = int(((pa != y) & (pb == y)).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"a_only_correct": b, "b_only_correct": c, "p_value": float(p)}


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    dfs, labels, info = load_protocol()
    n = len(labels)
    train_counts = dfs["train"]["label"].value_counts().reindex(labels).astype(int).tolist()

    experiments, rows = [], []
    for exp, rs in runs.items():
        family, weighted, extra = EXPERIMENTS[exp]
        metrics = {s: {m: ms([r["eval"][s][m] for r in rs]) for m in METRICS} for s in SPLITS}
        per_class = np.mean([r["eval"]["test_strict"]["per_class_f1"] for r in rs], axis=0)
        experiments.append({
            "exp": exp, "display": DISPLAY[exp], "family": family, "class_weighted": weighted,
            "hf_model": extra.get("hf"), "n_params": rs[0]["n_params"], "seeds": [r["seed"] for r in rs],
            "train_seconds": ms([r["train_seconds"] for r in rs]),
            "best_epoch": [r.get("best_epoch") for r in rs],
            "metrics": {s: {m: v["mean"] for m, v in d.items()} for s, d in metrics.items()},
            "metrics_std": {s: {m: v["std"] for m, v in d.items()} for s, d in metrics.items()},
            "per_class_f1_strict": per_class.tolist(),
        })
        rows.append({"Model": DISPLAY[exp], "Params": rs[0]["n_params"], "Seeds": len(rs),
                     **{f"{s} {m}": f"{metrics[s][m]['mean']:.4f} ± {metrics[s][m]['std']:.4f}"
                        for s in ("dev", "test", "test_strict", "climate_test")
                        for m in ("accuracy", "macro_f1")},
                     "Strict weighted F1": f"{metrics['test_strict']['weighted_f1']['mean']:.4f}",
                     "Train s/run": f"{np.mean([r['train_seconds'] for r in rs]):.0f}"})

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "final_results.csv", index=False)
    per_class = pd.DataFrame({e["display"]: e["per_class_f1_strict"] for e in experiments}, index=labels)
    per_class.insert(0, "train_count", train_counts)
    per_class.round(4).to_csv(RESULTS / "per_class_f1_strict.csv")

    # Model selection: highest mean dev macro-F1 (test never used for selection)
    learned = [e for e in experiments if e["family"] != "tfidf"]
    selected = max(learned, key=lambda e: e["metrics"]["dev"]["macro_f1"])
    by_exp = {e["exp"]: e for e in experiments}

    # Significance on strict test, seed 42 predictions: selected model vs every other
    # Paired tests need one run per model on the same rows: seed 42, else the first shared seed.
    common = sorted(set.intersection(*[{r["seed"] for r in rs} for rs in runs.values()]))
    ref_seed = 42 if 42 in common else common[0]
    s42 = {exp: next(r for r in rs if r["seed"] == ref_seed) for exp, rs in runs.items()}
    y = s42[selected["exp"]]["true"]["test_strict"]
    tests = []
    for exp in runs:
        if exp == selected["exp"]:
            continue
        pa, pb = s42[selected["exp"]]["pred"]["test_strict"], s42[exp]["pred"]["test_strict"]
        lo, hi = bootstrap_diff(y, pa, pb, n)
        tests.append({"against": exp, "display": DISPLAY[exp],
                      "delta_macro_f1_seed42": f1_score(y, pa, average="macro", labels=range(n), zero_division=0)
                      - f1_score(y, pb, average="macro", labels=range(n), zero_division=0),
                      "bootstrap_95ci": [lo, hi], **mcnemar(y, pa, pb)})
    pd.DataFrame(tests).round(4).to_csv(RESULTS / "significance.csv", index=False)

    cm = confusion_matrix(y, s42[selected["exp"]]["pred"]["test_strict"], labels=range(n))
    summary = {"labels": labels, "train_counts": train_counts, "protocol": info,
               "selected": selected["exp"], "experiments": experiments, "significance": tests,
               "reference_seed": ref_seed,
               "confusion_strict_seed42": {"exp": selected["exp"], "seed": ref_seed, "matrix": cm.tolist()}}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))

    _figures(experiments, by_exp, selected, labels, train_counts, cm, runs, ref_seed)
    print(table.to_string(index=False))
    print("\nselected by dev macro-F1:", selected["display"])
    print(pd.DataFrame(tests)[["display", "delta_macro_f1_seed42", "bootstrap_95ci", "p_value"]].to_string(index=False))


def _figures(experiments, by_exp, selected, labels, train_counts, cm, runs, ref_seed):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    names = [e["display"] for e in experiments]
    x = np.arange(len(experiments))

    # 1. In-domain vs transfer macro-F1 with seed std
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for k, (split, color, lab) in enumerate([("test_strict", "#2f5d8a", "LOGIC strict test"),
                                             ("climate_test", "#c0803a", "LOGICCLIMATE (transfer)")]):
        means = [e["metrics"][split]["macro_f1"] for e in experiments]
        stds = [e["metrics_std"][split]["macro_f1"] for e in experiments]
        ax.bar(x + (k - 0.5) * 0.38, means, 0.38, yerr=stds, capsize=2, color=color, label=lab)
    ax.set_xticks(x, names, rotation=35, ha="right")
    ax.set_ylabel(f"macro-F1 (mean ± std, {len(experiments[0]['seeds'])} seeds)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison.png", dpi=200)
    plt.close(fig)

    # 2. Per-class F1 heatmap
    order = np.argsort(train_counts)[::-1]
    mat = np.array([e["per_class_f1_strict"] for e in experiments])[:, order]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(names)), names)
    ax.set_xticks(range(len(labels)), [f"{labels[i]} ({train_counts[i]})" for i in order],
                  rotation=40, ha="right")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if mat[i, j] > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="F1 (strict test, mean of seeds)")
    fig.tight_layout()
    fig.savefig(FIGURES / "per_class_f1.png", dpi=200)
    plt.close(fig)

    # 3. Confusion matrix of the selected model (seed 42, strict test)
    fig, ax = plt.subplots(figsize=(7, 6))
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_title(f"{selected['display']} — strict LOGIC test (seed {ref_seed}), row-normalised")
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_matrix_selected.png", dpi=200)
    plt.close(fig)

    # 4. Effect of class weighting on per-class F1 vs train frequency
    pairs = [(a, a + "_cw") for a in ("cnn", "bilstm", "distilbert")
             if a in by_exp and a + "_cw" in by_exp]
    if pairs:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for a, b in pairs:
            d = np.array(by_exp[b]["per_class_f1_strict"]) - np.array(by_exp[a]["per_class_f1_strict"])
            ax.scatter(train_counts, d, s=18, label=by_exp[a]["display"])
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xlabel("training examples in class")
        ax.set_ylabel("ΔF1 (class-weighted − plain)")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES / "class_weighting_effect.png", dpi=200)
        plt.close(fig)

    # 5. Dev macro-F1 per epoch (seed 42) for learned models
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for exp, rs in runs.items():
        r = rs[0]
        if r.get("history") and not exp.endswith("_cw"):
            h = r["history"]
            ax.plot([p["epoch"] for p in h], [p["dev_macro_f1"] for p in h], marker="o", ms=2.5,
                    label=DISPLAY[exp])
    ax.set_xlabel("epoch")
    ax.set_ylabel("dev macro-F1")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "dev_curves.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
