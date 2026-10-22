"""Make measured tables, figures, uncertainty estimates, and error examples."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

import train_final as f


def main():
    res, figdir = f.RESULTS, f.FIGURES
    figdir.mkdir(parents=True, exist_ok=True)
    data, protocol = f.load_data()
    labels = protocol["labels"]
    tokenizer = f.AutoTokenizer.from_pretrained(f.MODEL_ID, revision=f.MODEL_REVISION)
    length_audit = {}
    for split in ("train", "dev", "test", "climate_test"):
        texts = data[split].text.tolist()
        length_audit[split] = {"rows": len(texts),
            "over_word_limit": sum(len(f.word_tokenize(t)) > f.WORD_MAX_LEN for t in texts),
            "over_transformer_limit": sum(len(tokenizer(t)["input_ids"]) > f.TRANSFORMER_MAX_LEN
                                          for t in texts)}
    (res / "length_audit.json").write_text(json.dumps(length_audit, indent=2) + "\n")
    models = ("cnn", "bilstm", "transformer", "transformer_weighted", "transformer_frozen",
              "transformer_tuned")
    metrics = json.loads((res / "metrics.json").read_text())
    winner = json.loads((res / "selection.json").read_text())["selected_by_dev"]
    comparison = pd.read_csv(res / "comparison.csv")

    # A trivial, transparent baseline on the same held-out rows.
    majority = data["train"].label.value_counts().idxmax()
    baseline = []
    for split in ("dev", "test", "test_exclusive", "climate_test"):
        d = data[split]
        y = d.y.to_numpy()
        pred = np.full(len(d), labels.index(majority))
        m = f.metrics(y, pred, labels)
        baseline.append({"model": "majority", "split": split, "n": len(d),
                         **{k: m[k] for k in ("accuracy", "macro_precision", "macro_recall",
                                                 "macro_f1", "weighted_f1")}})
    pd.concat([pd.DataFrame(baseline), comparison]).to_csv(res / "comparison_with_baseline.csv", index=False)

    plot = comparison[comparison.split.isin(["test_exclusive", "climate_test"])]
    pivot = plot.pivot(index="model", columns="split", values="macro_f1").loc[list(models)]
    ax = pivot.plot.bar(figsize=(10, 5), ylim=(0, max(0.6, pivot.max().max() + .05)), rot=20)
    ax.set_ylabel("Macro F1")
    ax.set_xlabel("")
    ax.set_title("Held-out LOGIC and direct LOGICCLIMATE transfer")
    ax.legend(["LOGICCLIMATE test", "LOGIC strict test"] if list(pivot.columns)[0] == "climate_test" else list(pivot.columns))
    plt.tight_layout()
    plt.savefig(figdir / "comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    for model in models:
        history = metrics[model]["history"]
        plt.plot([h["epoch"] for h in history], [h["dev_macro_f1"] for h in history],
                 marker="o", label=model)
    plt.xlabel("Epoch")
    plt.ylabel("Dev macro F1")
    plt.title("Validation learning curves (model selection uses dev only)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figdir / "learning_curves.png", dpi=160)
    plt.close()

    preds = pd.read_csv(res / f"{winner}_test_exclusive_predictions.csv")
    true = preds.label.map({x: i for i, x in enumerate(labels)}).to_numpy()
    predicted = preds.predicted.map({x: i for i, x in enumerate(labels)}).to_numpy()
    cm = confusion_matrix(true, predicted, labels=list(range(len(labels))))
    plt.figure(figsize=(11, 9))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(labels)), labels, rotation=70, ha="right", fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{winner}: strict LOGIC test confusion matrix (n={len(preds)})")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figdir / "confusion_matrix.png", dpi=160)
    plt.close()

    per_class = metrics[winner]["test_exclusive"]["per_class"]
    pd.DataFrame([{"label": k, **v} for k, v in per_class.items()]).to_csv(res / "selected_per_class.csv", index=False)
    errors = preds[~preds.correct].sort_values("confidence", ascending=False)
    errors.head(20).to_csv(res / "selected_errors.csv", index=False)

    # Paired bootstrap: same test rows and fixed 13-class label universe.
    runner_up = max((name for name in models if name != winner),
                    key=lambda name: metrics[name]["best_dev_macro_f1"])
    alternate = pd.read_csv(res / f"{runner_up}_test_exclusive_predictions.csv")
    assert preds.text.equals(alternate.text) and preds.label.equals(alternate.label)
    alternate_pred = alternate.predicted.map({x: i for i, x in enumerate(labels)}).to_numpy()
    rng = np.random.default_rng(f.SEED)
    boot_selected, boot_delta = [], []
    for _ in range(2000):
        ix = rng.integers(0, len(true), len(true))
        selected_f1 = f1_score(true[ix], predicted[ix], labels=list(range(len(labels))),
                               average="macro", zero_division=0)
        alternate_f1 = f1_score(true[ix], alternate_pred[ix], labels=list(range(len(labels))),
                                average="macro", zero_division=0)
        boot_selected.append(selected_f1)
        boot_delta.append(selected_f1 - alternate_f1)
    uncertainty = {"method": "2000 paired row bootstrap replicates, seed 42; fixed 13-label macro F1",
                   "selected_model": winner,
                   "runner_up_by_dev": runner_up,
                   "strict_test_macro_f1_95pct_interval": np.quantile(boot_selected, [.025, .975]).tolist(),
                   "selected_minus_runner_up_95pct_interval": np.quantile(boot_delta, [.025, .975]).tolist(),
                   "selected_minus_runner_up_point": float(metrics[winner]["test_exclusive"]["macro_f1"] -
                                                             metrics[runner_up]["test_exclusive"]["macro_f1"])}
    (res / "uncertainty.json").write_text(json.dumps(uncertainty, indent=2) + "\n")
    print("selected", winner)
    print(pd.read_csv(res / "comparison_with_baseline.csv").to_string(index=False))
    print("bootstrap", uncertainty)


if __name__ == "__main__":
    main()
