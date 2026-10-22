"""CLI demo for the completed FallacyLens experiment.

Usage: .venv/bin/python demo.py "Your argument here"
"""
import argparse
import json

import torch

import train_final as f


def predict(text: str) -> list[tuple[str, float]]:
    f.seed_everything()
    data, info = f.load_data()
    selection = json.loads((f.RESULTS / "selection.json").read_text())
    name = selection["selected_by_dev"]
    vocab = f.build_vocab(data["train"]) if not name.startswith("transformer") else None
    tokenizer = (f.AutoTokenizer.from_pretrained(f.MODEL_ID, revision=f.MODEL_REVISION)
                 if name.startswith("transformer") else None)
    if name == "cnn":
        model = f.CNN(len(vocab), len(info["labels"]))
    elif name == "bilstm":
        model = f.BiLSTM(len(vocab), len(info["labels"]))
    else:
        model = f.TinyBERT(len(info["labels"]), frozen=name == "transformer_frozen")
    model.load_state_dict(torch.load(f.RESULTS / f"{name}.pt", map_location="cpu", weights_only=True))
    model.eval()
    if name.startswith("transformer"):
        enc = tokenizer(f.normalize(text), return_tensors="pt", truncation=True,
                        padding="max_length", max_length=f.TRANSFORMER_MAX_LEN)
        args = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}
    else:
        tokens = f.word_tokenize(f.normalize(text))[:f.WORD_MAX_LEN]
        ids = [vocab.get(t, 1) for t in tokens]
        ids += [0] * (f.WORD_MAX_LEN - len(ids))
        args = {"input_ids": torch.tensor([ids])}
    with torch.no_grad():
        scores = torch.softmax(model(**args), 1)[0]
    top = torch.topk(scores, 3)
    return [(info["labels"][i], float(p)) for p, i in zip(top.values, top.indices)]


def main():
    parser = argparse.ArgumentParser(description="Classify an argument among 13 LOGIC fallacy labels")
    parser.add_argument("text", help="Argument text to classify")
    args = parser.parse_args()
    print("Top 3 predicted labels (model scores, not verified probabilities):")
    for label, score in predict(args.text):
        print(f"  {label:25s} {score:.3f}")
    print("This classifier cannot determine whether a claim is true or whether an argument is actually fallacious.")


if __name__ == "__main__":
    main()
