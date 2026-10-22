"""Export the measured TF-IDF baseline for dependency-free Vercel inference.

Run from the repository root with the experiment environment installed:
    python web/scripts/export_baseline.py

The export is checked against the committed seed-42 strict-test predictions.
"""

import json
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fallacylens.data import ROOT, load_descriptions, load_protocol


def write_module(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("export default " + json.dumps(value, ensure_ascii=False, separators=(",", ":")) + ";\n")


def main():
    splits, labels, _ = load_protocol()
    train = splits["train"]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    features = vec.fit_transform(train["text"])
    clf = LogisticRegression(max_iter=2000, C=10.0, random_state=42)
    clf.fit(features, train["y"])

    expected = json.loads((ROOT / "results/runs/tfidf_lr_s42.json").read_text())
    for split in ("dev", "test", "test_strict", "climate_test"):
        predictions = clf.predict(vec.transform(splits[split]["text"])).tolist()
        if predictions != expected["pred"][split]:
            raise ValueError(f"Exported baseline differs from the saved {split} run")

    descriptions = load_descriptions()
    model = {
        "labels": labels,
        "vocabulary": vec.vocabulary_,
        "idf": vec.idf_.tolist(),
        "coefficients": clf.coef_.tolist(),
        "intercepts": clf.intercept_.tolist(),
        "n_params": int(clf.coef_.size + clf.intercept_.size),
    }
    examples = splits["test_strict"].groupby("label", group_keys=False).head(2)
    out = ROOT / "web/lib"
    write_module(out / "model-data.js", model)
    write_module(out / "summary-data.js", json.loads((ROOT / "results/summary.json").read_text()))
    write_module(out / "examples-data.js", [
        {"text": row.text, "label": row.label} for row in examples.itertuples()
    ])
    write_module(out / "descriptions-data.js", {
        label: {"name": descriptions.get(label, (label, ""))[0],
                "description": descriptions.get(label, ("", ""))[1]}
        for label in labels
    })
    print(f"Exported {len(vec.vocabulary_)} TF-IDF features and {len(labels)} classes")


if __name__ == "__main__":
    main()
