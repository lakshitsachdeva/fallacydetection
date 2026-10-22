"""Inference + word-level occlusion explanations for every trained model."""
import json
import re

import numpy as np
import torch

from .data import ROOT, Vocab, normalise, tokenize
from .models import FallacyBiLSTM, FallacyCNN

MODELS_DIR = ROOT / "models"
_word_re = re.compile(r"\S+")


class Predictor:
    """Wraps one trained model behind a single `probs(texts) -> np.ndarray` call."""

    def __init__(self, path, device="cpu"):
        self.path = path
        self.meta = json.loads((path / "meta.json").read_text())
        self.labels = self.meta["labels"]
        self.family = self.meta["family"]
        self.device = torch.device(device)
        if self.family in ("cnn", "bilstm"):
            ckpt = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
            self.vocab = Vocab(ckpt["itos"])
            self.max_len = ckpt["max_len"]
            cls = FallacyCNN if self.family == "cnn" else FallacyBiLSTM
            self.model = cls(len(self.vocab), len(self.labels))
            self.model.load_state_dict(ckpt["state_dict"])
        else:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def probs(self, texts):
        texts = [normalise(t) for t in texts]
        out = []
        for i in range(0, len(texts), 64):
            chunk = texts[i:i + 64]
            if self.family in ("cnn", "bilstm"):
                x = torch.tensor([self.vocab.encode(t, self.max_len) for t in chunk],
                                 device=self.device)
                logits = self.model(x)
            else:
                x = self.tok(chunk, truncation=True, max_length=128, padding=True,
                             return_tensors="pt").to(self.device)
                logits = self.model(**x).logits
            out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        return np.concatenate(out)

    def _visible_words(self, text):
        words = _word_re.findall(text)
        for n in range(len(words), 0, -1):
            prefix = " ".join(words[:n])
            if self.family in ("cnn", "bilstm"):
                fits = len(tokenize(prefix)) <= self.max_len
            else:
                fits = len(self.tok(prefix, truncation=False)["input_ids"]) <= 128
            if fits:
                return words[:n]
        return words[:1]

    def explain(self, text, top_k=5):
        """Prediction plus each word's occlusion importance for the predicted class.

        Importance of word i = p(pred | text) - p(pred | text with word i removed).
        Positive values mean the word supports the prediction.
        """
        original = text
        # Score only the words inside the model's input window, so deletions never
        # pull unseen words in and truncated words are not shown as irrelevant.
        words = self._visible_words(normalise(text))
        text = " ".join(words)
        base = self.probs([text])[0]
        pred = int(base.argmax())
        variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
        imp = (base[pred] - self.probs(variants)[:, pred]) if variants else np.array([])
        order = np.argsort(-base)[:top_k]
        return {
            "label": self.labels[pred],
            "confidence": float(base[pred]),
            "top": [{"label": self.labels[i], "prob": float(base[i])} for i in order],
            "tokens": [{"text": w, "weight": float(v)} for w, v in zip(words, imp)],
            "truncated": len(words) < len(_word_re.findall(normalise(original))),
        }


def available_models():
    return sorted(p.name for p in MODELS_DIR.glob("*") if (p / "meta.json").exists())
