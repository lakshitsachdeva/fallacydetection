"""Occlusion probe: top-weighted words for correctly classified strict-test sentences per class."""
import sys
from collections import Counter

from .data import load_protocol
import torch

from .predictor import MODELS_DIR, Predictor

name = sys.argv[1] if len(sys.argv) > 1 else "distilbert"
p = Predictor(MODELS_DIR / name, device="cuda" if torch.cuda.is_available() else "cpu")
dfs, labels, _ = load_protocol()
test = dfs["test_strict"]
for lab in ("ad populum", "false dilemma", "ad hominem", "false causality", "circular reasoning"):
    words = Counter()
    for t in test[test["label"] == lab]["text"]:
        r = p.explain(t)
        if r["label"] != lab:
            continue
        top = sorted(r["tokens"], key=lambda x: -x["weight"])[:2]
        words.update(w["text"].lower().strip(".,!?\"'") for w in top)
    print(lab, words.most_common(8))
