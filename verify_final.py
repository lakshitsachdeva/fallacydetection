"""Check that the reported final metrics agree with saved predictions."""
import json

import numpy as np
import pandas as pd

import train_final as f


def main():
    root = f.RESULTS
    protocol = json.loads((root / "protocol.json").read_text())
    metrics = json.loads((root / "metrics.json").read_text())
    selection = json.loads((root / "selection.json").read_text())["selected_by_dev"]
    labels = protocol["labels"]
    assert selection == max(metrics, key=lambda name: metrics[name]["best_dev_macro_f1"])
    rows = pd.read_csv(root / "comparison.csv")
    for name, record in metrics.items():
        for split in ("dev", "test", "test_exclusive", "climate_test"):
            pred = pd.read_csv(root / f"{name}_{split}_predictions.csv")
            assert len(pred) == protocol["final_rows"][split]
            assert pred.label.isin(labels).all() and pred.predicted.isin(labels).all()
            assert np.array_equal(pred.correct.to_numpy(), (pred.label == pred.predicted).to_numpy())
            mapping = {label: i for i, label in enumerate(labels)}
            measured = f.metrics(pred.label.map(mapping), pred.predicted.map(mapping), labels)
            for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"):
                assert abs(measured[key] - record[split][key]) < 1e-10, (name, split, key)
                row = rows[(rows.model == name) & (rows.split == split)]
                assert len(row) == 1 and abs(measured[key] - float(row.iloc[0][key])) < 1e-10
    assert len(pd.read_csv(root / "token_sensitivity.csv")) == 10
    assert (f.FIGURES / "comparison.png").exists()
    print(f"Verified {len(metrics)} models, {len(rows)} result rows, all predictions, and dev selection: {selection}")


if __name__ == "__main__":
    main()
