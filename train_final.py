"""Reproducible FallacyLens Phase 2/3 experiments.

Run from the repository root: .venv/bin/python train_final.py
The official CSVs and pretrained model are downloaded on first use.
"""
from __future__ import annotations

import copy
import argparse
import hashlib
import json
import os
import random
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results" / "final"
FIGURES = ROOT / "figures" / "final"
DATA_REF = "2a3d09708122c0994279df034d867b80950700ea"
BASE = f"https://raw.githubusercontent.com/causalNLP/logical-fallacy/{DATA_REF}/data/"
EXPECTED_HASHES = {
    "edu_train.csv": "0c6377dbf744ba85e60102eb9842cc770f385afcb2724f607e8e8a99f5d9545c",
    "edu_dev.csv": "5c567eb12e604271d0ec10e456c5c0bedcf01bd8a2f56d84a5ca88744ece36af",
    "edu_test.csv": "47f6b86ec6c5256e9db97844f86684dc1fdc72c593e14db230e532098f668534",
    "climate_test.csv": "16f296fb38e594f92512a8f05e6b8c526a978fafe9fce8f924faaf7ad7d3082a",
}
MODEL_ID = "prajjwal1/bert-tiny"
MODEL_REVISION = "6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837"
SEED = 42
WORD_MAX_LEN = 96
TRANSFORMER_MAX_LEN = 128
EPOCHS = {"cnn": 16, "bilstm": 16, "transformer": 8, "transformer_weighted": 8,
          "transformer_frozen": 8, "transformer_tuned": 12}
PATIENCE = 3
BATCH = 32
DEVICE = torch.device("cpu")  # Stable across Mac and Colab; tiny model is CPU-feasible.
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[a-z]+)?|[^\w\s]")


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def normalize(s: object) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def key(s: object) -> str:
    return normalize(s).casefold()


def download() -> dict[str, str]:
    RAW.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in ("edu_train", "edu_dev", "edu_test", "climate_test"):
        path = RAW / f"{name}.csv"
        if not path.exists():
            urllib.request.urlretrieve(BASE + path.name, path)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashes[path.name] != EXPECTED_HASHES[path.name]:
            raise ValueError(f"Unexpected official dataset hash: {path.name}")
    return hashes


def load_data() -> tuple[dict[str, pd.DataFrame], dict]:
    hashes = download()
    dfs = {}
    for split in ("train", "dev", "test"):
        raw = pd.read_csv(RAW / f"edu_{split}.csv")
        df = raw[["source_article", "updated_label"]].rename(
            columns={"source_article": "text", "updated_label": "label"}).dropna().copy()
        df["text"] = df.text.map(normalize)
        df = df[df.text.ne("")].reset_index(drop=True)
        df["key"] = df.text.map(key)
        dfs[split] = df
    original = {s: len(df) for s, df in dfs.items()}
    # Conflicting labels for identical text cannot be resolved from the text.
    conflicts = dfs["train"].groupby("key").label.nunique()
    bad = set(conflicts[conflicts > 1].index)
    train = dfs["train"][~dfs["train"].key.isin(bad)].copy()
    after_conflicts = len(train)
    train = train.drop_duplicates("key", keep="first").copy()
    after_dedup = len(train)
    # Remove training examples seen in dev; do not use test examples to edit
    # training data. The strict test subset below excludes seen text instead.
    heldout_keys = set(dfs["dev"].key)
    train = train[~train.key.isin(heldout_keys)].reset_index(drop=True)
    dfs["train"] = train
    labels = sorted(set(train.label) | set(dfs["dev"].label) | set(dfs["test"].label))
    label2id = {label: i for i, label in enumerate(labels)}
    for s, df in dfs.items():
        df["y"] = df.label.map(label2id).astype(int)
    # Keep the official test intact, but also report a test subset with no
    # text seen in dev during early stopping/model selection.
    seen_keys = set(train.key) | set(dfs["dev"].key)
    dfs["test_exclusive"] = dfs["test"][~dfs["test"].key.isin(seen_keys)].copy()
    climate = pd.read_csv(RAW / "climate_test.csv")
    climate = climate[["source_article", "logical_fallacies"]].rename(
        columns={"source_article": "text", "logical_fallacies": "label"}).dropna().copy()
    climate["text"] = climate.text.map(normalize)
    climate = climate[climate.text.ne("")].reset_index(drop=True)
    climate["key"] = climate.text.map(key)
    climate = climate[climate.label.isin(label2id)].copy()
    climate["y"] = climate.label.map(label2id).astype(int)
    dfs["climate_test"] = climate
    assert not (set(train.key) & set(dfs["dev"].key))
    assert not (set(dfs["test_exclusive"].key) & seen_keys)
    info = {"source": BASE, "sha256": hashes, "original_rows": original,
            "conflicting_train_texts": len(bad), "train_after_conflicts": after_conflicts,
            "train_after_dedup": after_dedup, "final_rows": {s: len(d) for s, d in dfs.items()},
            "train_test_overlap_rows": int(dfs["test"].key.isin(set(train.key)).sum()),
            "dev_test_overlap_rows": int(dfs["test"].key.isin(set(dfs["dev"].key)).sum()),
            "test_rows_excluded_for_prior_text_exposure": len(dfs["test"]) - len(dfs["test_exclusive"]),
            "labels": labels, "seed": SEED}
    return dfs, info


def word_tokenize(s: str) -> list[str]:
    return TOKEN_RE.findall(s.lower())


def build_vocab(train: pd.DataFrame) -> dict[str, int]:
    counts = Counter(tok for txt in train.text for tok in word_tokenize(txt))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, count in counts.items():
        if count >= 2:
            vocab[token] = len(vocab)
    return vocab


class TensorDataset(Dataset):
    def __init__(self, enc: dict[str, torch.Tensor], labels: pd.Series):
        self.enc = enc
        self.labels = torch.tensor(labels.to_numpy(), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {**{k: v[i] for k, v in self.enc.items()}, "labels": self.labels[i]}


class CNN(nn.Module):
    def __init__(self, vocab: int, n: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, 100, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(100, 100, k) for k in (3, 4, 5)])
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(300, n)

    def forward(self, input_ids, attention_mask=None):
        emb = self.embedding(input_ids).transpose(1, 2)
        features = [torch.relu(c(emb)).max(2).values for c in self.convs]
        return self.out(self.dropout(torch.cat(features, dim=1)))


class BiLSTM(nn.Module):
    def __init__(self, vocab: int, n: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, 100, padding_idx=0)
        self.lstm = nn.LSTM(100, 96, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(192, n)

    def forward(self, input_ids, attention_mask=None):
        lengths = input_ids.ne(0).sum(1).clamp(min=1).cpu()
        emb = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True,
                                                   enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.out(self.dropout(torch.cat([h[-2], h[-1]], dim=1)))


class TinyBERT(nn.Module):
    def __init__(self, n: int, frozen: bool = False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        if frozen:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(self.encoder.config.hidden_size, n)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.out(self.dropout(hidden[:, 0]))


def make_datasets(dfs: dict[str, pd.DataFrame]):
    vocab = build_vocab(dfs["train"])
    def encode_words(texts):
        arr = np.zeros((len(texts), WORD_MAX_LEN), dtype=np.int64)
        for i, txt in enumerate(texts):
            ids = [vocab.get(t, 1) for t in word_tokenize(txt)[:WORD_MAX_LEN]]
            arr[i, :len(ids)] = ids
        return {"input_ids": torch.from_numpy(arr)}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    word_sets = {s: TensorDataset(encode_words(df.text), df.y) for s, df in dfs.items()}
    bert_sets = {s: TensorDataset({k: torch.tensor(v) for k, v in tokenizer(
        df.text.tolist(), truncation=True, padding="max_length", max_length=TRANSFORMER_MAX_LEN).items()
        if k in ("input_ids", "attention_mask")}, df.y) for s, df in dfs.items()}
    return word_sets, bert_sets, vocab, tokenizer


def loader(ds: Dataset, shuffle: bool = False) -> DataLoader:
    g = torch.Generator().manual_seed(SEED)
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, generator=g)


def predict(model, ds: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys, preds, probs = [], [], []
    with torch.no_grad():
        for batch in loader(ds):
            y = batch.pop("labels")
            p = torch.softmax(model(**batch), dim=1)
            ys.extend(y.tolist())
            preds.extend(p.argmax(1).tolist())
            probs.extend(p.tolist())
    return np.asarray(ys), np.asarray(preds), np.asarray(probs)


def metrics(y, pred, labels):
    p, r, f, support = precision_recall_fscore_support(
        y, pred, labels=list(range(len(labels))), average=None, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(
        y, pred, labels=list(range(len(labels))), average="macro", zero_division=0)
    _, _, fw, _ = precision_recall_fscore_support(
        y, pred, labels=list(range(len(labels))), average="weighted", zero_division=0)
    return {"accuracy": float(accuracy_score(y, pred)), "macro_precision": float(pm),
            "macro_recall": float(rm), "macro_f1": float(fm), "weighted_f1": float(fw),
            "per_class": {name: {"precision": float(p[i]), "recall": float(r[i]),
                                   "f1": float(f[i]), "support": int(support[i])}
                          for i, name in enumerate(labels)}}


def train(name, datasets, labels, class_counts):
    seed_everything()
    if name == "cnn":
        model = CNN(len(VOCAB), len(labels))
    elif name == "bilstm":
        model = BiLSTM(len(VOCAB), len(labels))
    else:
        model = TinyBERT(len(labels), frozen=name == "transformer_frozen")
    model.to(DEVICE)
    lr = (1e-4 if name == "transformer_tuned" else
          2e-5 if name.startswith("transformer") and name != "transformer_frozen" else 1e-3)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    if name == "transformer_weighted":
        weights = np.sqrt(class_counts.max() / class_counts)
        weights = weights / weights.mean()
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    else:
        loss_fn = nn.CrossEntropyLoss()
    best_f1, best_state, best_epoch, bad_epochs = -1., None, 0, 0
    history = []
    start = time.time()
    for epoch in range(1, EPOCHS[name] + 1):
        model.train()
        total_loss, seen = 0., 0
        for batch in loader(datasets["train"], shuffle=True):
            y = batch.pop("labels")
            optimizer.zero_grad()
            logits = model(**batch)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total_loss += loss.item() * len(y)
            seen += len(y)
        dy, dp, _ = predict(model, datasets["dev"])
        dev_f1 = metrics(dy, dp, labels)["macro_f1"]
        history.append({"epoch": epoch, "train_loss": total_loss / seen,
                        "dev_macro_f1": dev_f1})
        print(f"{name:21s} epoch {epoch:02d} loss={total_loss/seen:.4f} dev_f1={dev_f1:.4f}", flush=True)
        if dev_f1 > best_f1 + 1e-9:
            best_f1, best_epoch, bad_epochs = dev_f1, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break
    model.load_state_dict(best_state)
    artifact = {"model": name, "best_epoch": best_epoch, "best_dev_macro_f1": best_f1,
                "seconds": round(time.time() - start, 1), "history": history,
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "learning_rate": lr, "epochs_max": EPOCHS[name], "patience": PATIENCE,
                "batch_size": BATCH}
    torch.save(model.state_dict(), RESULTS / f"{name}.pt")
    return model, artifact


def explain(model, ds, df, labels, tokenizer, name):
    """Deletion-based local sensitivity for five correctly and five wrongly predicted rows."""
    y, pred, probs = predict(model, ds)
    chosen = list(np.where(y != pred)[0][:5]) + list(np.where(y == pred)[0][:5])
    rows = []
    for idx in chosen:
        record = ds[idx]
        ids = record["input_ids"].clone()
        mask = record.get("attention_mask", ids.ne(0).long())
        valid = int(mask.sum())
        base = float(probs[idx, pred[idx]])
        drops = []
        for pos in range(1 if name.startswith("transformer") else 0, min(valid - 1, 40)):
            modified = ids.clone()
            modified[pos] = tokenizer.mask_token_id if name.startswith("transformer") else 1
            with torch.no_grad():
                score = torch.softmax(model(input_ids=modified[None, :],
                    **({"attention_mask": mask[None, :]} if name.startswith("transformer") else {})), 1)
            drops.append((pos, base - float(score[0, pred[idx]])))
        top = sorted(drops, key=lambda x: x[1], reverse=True)[:5]
        token_names = tokenizer.convert_ids_to_tokens(ids.tolist()) if name.startswith("transformer") else \
            [INVERSE_VOCAB.get(int(x), "<UNK>") for x in ids]
        rows.append({"row": int(idx), "correct": bool(y[idx] == pred[idx]),
                     "true": labels[y[idx]], "pred": labels[pred[idx]],
                     "confidence": base, "text": df.iloc[idx].text,
                     "top_deletion_tokens": "; ".join(f"{token_names[pos]} ({drop:+.3f})" for pos, drop in top)})
    pd.DataFrame(rows).to_csv(RESULTS / "token_sensitivity.csv", index=False)


def main():
    global VOCAB, INVERSE_VOCAB
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="skip complete saved experiments")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    seed_everything()
    dfs, info = load_data()
    labels = info["labels"]
    word_sets, bert_sets, VOCAB, tokenizer = make_datasets(dfs)
    INVERSE_VOCAB = {v: k for k, v in VOCAB.items()}
    info.update({"vocab_size": len(VOCAB), "word_max_len": WORD_MAX_LEN,
                 "transformer_max_len": TRANSFORMER_MAX_LEN, "pretrained_model": MODEL_ID,
                 "pretrained_revision": MODEL_REVISION,
                 "torch_version": torch.__version__})
    (RESULTS / "protocol.json").write_text(json.dumps(info, indent=2) + "\n")
    counts = dfs["train"].y.value_counts().reindex(range(len(labels)), fill_value=0).to_numpy()
    all_results = json.loads((RESULTS / "metrics.json").read_text()) if args.resume and (RESULTS / "metrics.json").exists() else {}
    models = ("cnn", "bilstm", "transformer", "transformer_weighted", "transformer_frozen",
              "transformer_tuned")
    for name in models:
        complete = name in all_results and (RESULTS / f"{name}.pt").exists() and all(
            (RESULTS / f"{name}_{s}_predictions.csv").exists()
            for s in ("dev", "test", "test_exclusive", "climate_test"))
        if args.resume and complete:
            print(f"{name}: using saved complete experiment", flush=True)
            continue
        ds = bert_sets if name.startswith("transformer") else word_sets
        model, record = train(name, ds, labels, counts)
        for split in ("dev", "test", "test_exclusive", "climate_test"):
            y, pred, probs = predict(model, ds[split])
            record[split] = metrics(y, pred, labels)
            frame = dfs[split][["text", "label"]].copy()
            frame["predicted"] = [labels[i] for i in pred]
            frame["confidence"] = probs.max(1)
            frame["correct"] = y == pred
            frame.to_csv(RESULTS / f"{name}_{split}_predictions.csv", index=False)
        all_results[name] = record
        (RESULTS / "metrics.json").write_text(json.dumps(all_results, indent=2) + "\n")
        print(f"{name} test macro-F1={record['test']['macro_f1']:.4f} climate={record['climate_test']['macro_f1']:.4f}", flush=True)
        del model
    # Model choice is based only on dev. Ties favor the cheaper model above.
    winner = max(models, key=lambda n: all_results[n]["best_dev_macro_f1"])
    ds = bert_sets if winner.startswith("transformer") else word_sets
    seed_everything()
    if winner == "cnn": model = CNN(len(VOCAB), len(labels))
    elif winner == "bilstm": model = BiLSTM(len(VOCAB), len(labels))
    else: model = TinyBERT(len(labels), frozen=winner == "transformer_frozen")
    model.load_state_dict(torch.load(RESULTS / f"{winner}.pt", map_location="cpu", weights_only=True))
    explain(model, ds["test_exclusive"], dfs["test_exclusive"], labels, tokenizer, winner)
    table = []
    for name in models:
        for split in ("dev", "test", "test_exclusive", "climate_test"):
            m = all_results[name][split]
            table.append({"model": name, "split": split, "n": len(dfs[split]),
                          **{k: m[k] for k in ("accuracy", "macro_precision", "macro_recall",
                                                   "macro_f1", "weighted_f1")}})
    pd.DataFrame(table).to_csv(RESULTS / "comparison.csv", index=False)
    (RESULTS / "selection.json").write_text(json.dumps({"selected_by_dev": winner,
        "dev_macro_f1": all_results[winner]["best_dev_macro_f1"]}, indent=2) + "\n")
    print("selected by dev:", winner, flush=True)


if __name__ == "__main__":
    main()
