"""Dataset loading and the leakage-controlled data protocol.

The protocol is the one introduced in the final-experiment PR (train_final.py):
pinned dataset commit with SHA-256 checks, removal of train texts whose
duplicates carry conflicting labels, train deduplication, removal of train
texts that also appear in dev, and a strict test subset that excludes any
test text already seen in train or dev. Tokenisation is the Phase 1
punctuation-preserving scheme with no stopword removal or stemming.
"""
import hashlib
import re
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_REF = "2a3d09708122c0994279df034d867b80950700ea"
BASE_URL = f"https://raw.githubusercontent.com/causalNLP/logical-fallacy/{DATA_REF}/data/"
EXPECTED_HASHES = {
    "edu_train.csv": "0c6377dbf744ba85e60102eb9842cc770f385afcb2724f607e8e8a99f5d9545c",
    "edu_dev.csv": "5c567eb12e604271d0ec10e456c5c0bedcf01bd8a2f56d84a5ca88744ece36af",
    "edu_test.csv": "47f6b86ec6c5256e9db97844f86684dc1fdc72c593e14db230e532098f668534",
    "climate_test.csv": "16f296fb38e594f92512a8f05e6b8c526a978fafe9fce8f924faaf7ad7d3082a",
}

_ws = re.compile(r"\s+")
_token_re = re.compile(r"[A-Za-z0-9]+(?:'[a-z]+)?|[^\w\s]")


def normalise(text):
    return _ws.sub(" ", str(text)).strip()


def text_key(text):
    return normalise(text).casefold()


def tokenize(text):
    return _token_re.findall(text.lower())


def _fetch(fname, check=True):
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    dest = DATA_RAW / fname
    if not dest.exists():
        urllib.request.urlretrieve(BASE_URL + fname, dest)
    if check and fname in EXPECTED_HASHES:
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        if digest != EXPECTED_HASHES[fname]:
            raise ValueError(f"{fname}: sha256 {digest} does not match the pinned release")
    return dest


def _read(fname, text_col, label_col):
    df = pd.read_csv(_fetch(fname))[[text_col, label_col]]
    df = df.rename(columns={text_col: "text", label_col: "label"}).dropna().copy()
    df["text"] = df["text"].map(normalise)
    df["label"] = df["label"].str.strip()
    df = df[df["text"].ne("")].reset_index(drop=True)
    df["key"] = df["text"].map(text_key)
    return df


def load_protocol():
    """Return ({split: DataFrame[text, label, key, y]}, labels, info).

    Splits: train, dev, test (official), test_strict, climate_test.
    """
    dfs = {s: _read(f"edu_{s}.csv", "source_article", "updated_label")
           for s in ("train", "dev", "test")}
    original = {s: len(d) for s, d in dfs.items()}

    train = dfs["train"]
    n_labels = train.groupby("key")["label"].nunique()
    conflicting = set(n_labels[n_labels > 1].index)
    train = train[~train["key"].isin(conflicting)]
    after_conflicts = len(train)
    train = train.drop_duplicates("key", keep="first")
    after_dedup = len(train)
    train = train[~train["key"].isin(set(dfs["dev"]["key"]))].reset_index(drop=True)
    dfs["train"] = train

    seen = set(train["key"]) | set(dfs["dev"]["key"])
    dfs["test_strict"] = dfs["test"][~dfs["test"]["key"].isin(seen)].reset_index(drop=True)

    labels = sorted(set(train["label"]) | set(dfs["dev"]["label"]) | set(dfs["test"]["label"]))
    climate = _read("climate_test.csv", "source_article", "logical_fallacies")
    dfs["climate_test"] = climate[climate["label"].isin(labels)].reset_index(drop=True)

    l2i = {l: i for i, l in enumerate(labels)}
    for d in dfs.values():
        d["y"] = d["label"].map(l2i).astype(int)

    info = {"source": BASE_URL, "original_rows": original,
            "conflicting_train_texts": len(conflicting),
            "train_after_conflicts": after_conflicts, "train_after_dedup": after_dedup,
            "final_rows": {s: len(d) for s, d in dfs.items()},
            "test_rows_seen_in_train_or_dev": len(dfs["test"]) - len(dfs["test_strict"])}
    return dfs, labels, info


def load_descriptions():
    """Label -> (readable name, description) from the dataset's own mappings.csv."""
    df = pd.read_csv(_fetch("mappings.csv", check=False))
    return {r["Original Name"].strip(): (r["Understandable Name"].strip(), str(r["Description"]).strip())
            for _, r in df.iterrows()}


class Vocab:
    PAD, UNK = 0, 1

    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build(cls, texts, min_freq=2):
        freq = Counter(tok for t in texts for tok in tokenize(t))
        itos = ["<PAD>", "<UNK>"] + [tok for tok, c in freq.items() if c >= min_freq]
        return cls(itos)

    def __len__(self):
        return len(self.itos)

    def encode(self, text, max_len):
        ids = [self.stoi.get(tok, self.UNK) for tok in tokenize(text)][:max_len]
        return ids + [self.PAD] * (max_len - len(ids))
