"""Phase 2/3 experiment runner.

Trains every model family under one protocol (same splits, preprocessing,
seeds, model selection on dev macro-F1, test labels never used for fitting or
selection) and evaluates each trained model on the official LOGIC test, the
strict LOGIC test (no text seen in train/dev) and, by direct transfer, on
LOGICCLIMATE test.

Usage:
    python -m fallacylens.train                       # all experiments, seeds 42 43 44
    python -m fallacylens.train --exp cnn bilstm --seeds 42
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

from .data import ROOT, Vocab, load_protocol
from .models import FallacyBiLSTM, FallacyCNN, build_transformer

RUNS_DIR = ROOT / "results" / "runs"
MODELS_DIR = ROOT / "models"
MAX_LEN = 54          # 95th percentile of train word length (Phase 1)
TF_MAX_LEN = 128
EVAL_SPLITS = ("dev", "test", "test_strict", "climate_test")

EXPERIMENTS = {
    # name: (family, class_weighted, extra)
    "tfidf_lr":       ("tfidf", False, {}),
    "tfidf_lr_cw":    ("tfidf", True, {}),
    "cnn":            ("cnn", False, {}),
    "cnn_cw":         ("cnn", True, {}),
    "bilstm":         ("bilstm", False, {}),
    "bilstm_cw":      ("bilstm", True, {}),
    "distilbert":     ("transformer", False, {"hf": "distilbert-base-uncased", "lr": 5e-5}),
    "distilbert_cw":  ("transformer", True, {"hf": "distilbert-base-uncased", "lr": 5e-5}),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics(y_true, y_pred, n_classes):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, wf, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    _, _, pcf, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": p, "recall": r,
            "macro_f1": f, "weighted_f1": wf, "per_class_f1": pcf.tolist(),
            "support": sup.tolist()}


def class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    return torch.tensor(len(y) / (n_classes * np.maximum(counts, 1)), dtype=torch.float)


# ---------------------------------------------------------------- TF-IDF + LR
def run_tfidf(data, labels, weighted, seed, **_):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    xtr = vec.fit_transform(data["train"]["text"])
    clf = LogisticRegression(max_iter=2000, C=10.0, random_state=seed,
                             class_weight="balanced" if weighted else None)
    clf.fit(xtr, data["train"]["y"])
    preds = {s: clf.predict(vec.transform(d["text"])).tolist() for s, d in data.items()}
    return preds, {"n_params": int(clf.coef_.size + clf.intercept_.size)}, None, None


# ---------------------------------------------------------------- CNN / BiLSTM
def run_neural(data, labels, weighted, seed, family, device, save_dir=None, **_):
    vocab = Vocab.build(data["train"]["text"], min_freq=2)
    enc = {s: torch.tensor([vocab.encode(t, MAX_LEN) for t in d["text"]], dtype=torch.long)
           for s, d in data.items()}
    ys = {s: torch.tensor(d["y"].values, dtype=torch.long) for s, d in data.items()}
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(TensorDataset(enc["train"], ys["train"]), batch_size=32,
                              shuffle=True, generator=g)
    n = len(labels)
    model = (FallacyCNN if family == "cnn" else FallacyBiLSTM)(len(vocab), n).to(device)
    weight = class_weights(data["train"]["y"].values, n).to(device) if weighted else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def predict(x):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 256):
                out.extend(model(x[i:i + 256].to(device)).argmax(1).cpu().tolist())
        return out

    history, best, best_state, best_epoch, patience = [], -1.0, None, 0, 4
    for epoch in range(1, 21):
        model.train()
        tot = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
        dev_f1 = metrics(ys["dev"].tolist(), predict(enc["dev"]), n)["macro_f1"]
        history.append({"epoch": epoch, "train_loss": tot / len(ys["train"]), "dev_macro_f1": dev_f1})
        if dev_f1 > best:
            best, best_epoch, patience = dev_f1, epoch, 4
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience -= 1
            if patience == 0:
                break
    model.load_state_dict(best_state)
    preds = {s: predict(enc[s]) for s in data}
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"family": family, "state_dict": model.state_dict(), "itos": vocab.itos,
                    "labels": labels, "max_len": MAX_LEN, "seed": seed},
                   save_dir / "model.pt")
    info = {"n_params": sum(p.numel() for p in model.parameters()), "best_epoch": best_epoch,
            "best_dev_macro_f1": best, "vocab_size": len(vocab)}
    return preds, info, history, model


# ---------------------------------------------------------------- Transformers
def run_transformer(data, labels, weighted, seed, device, hf, lr, save_dir=None, **_):
    from transformers import get_linear_schedule_with_warmup
    tok, model = build_transformer(hf, len(labels), labels)
    model.to(device)
    n = len(labels)

    def batches(df, bs, shuffle, rng=None):
        idx = np.arange(len(df))
        if shuffle:
            rng.shuffle(idx)
        texts, y = df["text"].tolist(), df["y"].values
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            x = tok([texts[j] for j in b], truncation=True, max_length=TF_MAX_LEN,
                    padding=True, return_tensors="pt")
            yield {k: v.to(device) for k, v in x.items()}, torch.tensor(y[b], device=device)

    def predict(df):
        model.eval()
        out = []
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            for x, _ in batches(df, 64, False):
                out.extend(model(**x).logits.argmax(1).cpu().tolist())
        return out

    epochs, bs = 8, 16
    steps = epochs * int(np.ceil(len(data["train"]) / bs))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    weight = class_weights(data["train"]["y"].values, n).to(device) if weighted else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    rng = np.random.RandomState(seed)

    history, best, best_state, best_epoch = [], -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        tot, cnt = 0.0, 0
        for x, yb in batches(data["train"], bs, True, rng):
            opt.zero_grad()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(**x).logits.float(), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(yb)
            cnt += len(yb)
        dev_f1 = metrics(data["dev"]["y"].tolist(), predict(data["dev"]), n)["macro_f1"]
        history.append({"epoch": epoch, "train_loss": tot / cnt, "dev_macro_f1": dev_f1})
        print(f"    epoch {epoch} loss {tot / cnt:.4f} dev macro-F1 {dev_f1:.4f}", flush=True)
        if dev_f1 > best:
            best, best_epoch = dev_f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    preds = {s: predict(d) for s, d in data.items()}
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
    info = {"n_params": sum(p.numel() for p in model.parameters()), "best_epoch": best_epoch,
            "best_dev_macro_f1": best, "hf_model": hf, "lr": lr}
    return preds, info, history, model


RUNNERS = {"tfidf": run_tfidf, "cnn": run_neural, "bilstm": run_neural, "transformer": run_transformer}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=list(EXPERIMENTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--save-seed", type=int, default=42, help="seed whose model is kept for the app")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, labels, info = load_protocol()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "protocol.json").write_text(json.dumps({**info, "labels": labels}, indent=1))
    print(f"device={device} | rows {info['final_rows']}")

    for exp in args.exp:
        family, weighted, extra = EXPERIMENTS[exp]
        for seed in args.seeds:
            out = RUNS_DIR / f"{exp}_s{seed}.json"
            needs_model = (seed == args.save_seed and family != "tfidf"
                           and not (MODELS_DIR / exp / "meta.json").exists())
            if out.exists() and not needs_model:
                print(f"skip {exp} seed {seed} (exists)")
                continue
            print(f"== {exp} seed {seed}", flush=True)
            set_seed(seed)
            save_dir = MODELS_DIR / exp if seed == args.save_seed and family != "tfidf" else None
            t0 = time.time()
            preds, info, history, _ = RUNNERS[family](
                data, labels, weighted, seed, family=family, device=device,
                save_dir=save_dir, **extra)
            elapsed = time.time() - t0
            record = {"exp": exp, "family": family, "class_weighted": weighted, "seed": seed,
                      "labels": labels, "train_seconds": elapsed, "device": str(device), **info,
                      "history": history,
                      "eval": {split: metrics(data[split]["y"].tolist(), preds[split], len(labels))
                               for split in EVAL_SPLITS},
                      "pred": {split: preds[split] for split in EVAL_SPLITS},
                      "true": {split: data[split]["y"].tolist() for split in EVAL_SPLITS}}
            out.write_text(json.dumps(record, indent=1))
            ev = record["eval"]
            print(f"   strict acc {ev['test_strict']['accuracy']:.4f} macro-F1 "
                  f"{ev['test_strict']['macro_f1']:.4f} | climate macro-F1 "
                  f"{ev['climate_test']['macro_f1']:.4f} | {elapsed:.0f}s", flush=True)
            if save_dir:
                (save_dir / "meta.json").write_text(json.dumps(
                    {"exp": exp, "family": family, "class_weighted": weighted, "seed": seed,
                     "labels": labels, "strict_test_macro_f1": ev["test_strict"]["macro_f1"],
                     "strict_test_accuracy": ev["test_strict"]["accuracy"], **info}, indent=1))


if __name__ == "__main__":
    main()
