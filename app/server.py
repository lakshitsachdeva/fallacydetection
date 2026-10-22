"""FallacyLens web API + static frontend.

Run from the repository root:
    python -m uvicorn app.server:app --port 8000
Then open http://127.0.0.1:8000 (after `npm run build` in web/), or run the
Vite dev server in web/ which proxies /api to this process.
"""
import json
from functools import lru_cache

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from fallacylens.data import ROOT, load_descriptions, load_protocol
from fallacylens.predictor import MODELS_DIR, Predictor, available_models

SUMMARY = ROOT / "results" / "summary.json"
DIST = ROOT / "web" / "dist"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CHARS = 2000

app = FastAPI(title="FallacyLens")


class TextIn(BaseModel):
    text: str = Field(min_length=3, max_length=MAX_CHARS)
    model: str | None = None


@lru_cache(maxsize=None)
def predictor(name):
    if name not in available_models():
        raise HTTPException(404, f"unknown model '{name}'")
    return Predictor(MODELS_DIR / name, device=DEVICE)


_summary_cache = {"mtime": None, "data": {}}


def summary():
    """results/summary.json, re-read whenever fallacylens.analyze rewrites it."""
    mtime = SUMMARY.stat().st_mtime if SUMMARY.exists() else None
    if mtime != _summary_cache["mtime"]:
        _summary_cache.update(mtime=mtime, data=json.loads(SUMMARY.read_text()) if mtime else {})
    return _summary_cache["data"]


@lru_cache(maxsize=1)
def descriptions():
    return load_descriptions()


@lru_cache(maxsize=None)
def model_meta(name):
    return json.loads((MODELS_DIR / name / "meta.json").read_text())


@lru_cache(maxsize=1)
def examples():
    dfs, _, _ = load_protocol()
    test = dfs["test_strict"]
    picked = test.groupby("label", group_keys=False).head(2)
    return [{"text": r.text, "label": r.label} for r in picked.itertuples()]


def default_model():
    names = available_models()
    selected = summary().get("selected")
    return selected if selected in names else (names[0] if names else None)


@app.get("/api/meta")
def meta():
    desc = descriptions()
    s = summary()
    exps = {e["exp"]: e for e in s.get("experiments", [])}
    models = []
    for name in available_models():
        m = model_meta(name)
        e = exps.get(name, {})
        models.append({"name": name, "display": e.get("display", name), "family": m["family"],
                       "class_weighted": m["class_weighted"], "n_params": m["n_params"],
                       "strict_macro_f1": e.get("metrics", {}).get("test_strict", {}).get("macro_f1")})
    labels = s.get("labels") or (model_meta(models[0]["name"])["labels"] if models else [])
    return {"default_model": default_model(), "models": models, "device": DEVICE,
            "labels": [{"id": l, "name": desc.get(l, (l, ""))[0], "description": desc.get(l, ("", ""))[1]}
                       for l in labels]}


@app.post("/api/analyze")
def analyze(body: TextIn):
    name = body.model or default_model()
    if name is None:
        raise HTTPException(503, "no trained models found; run python -m fallacylens.train first")
    return {"model": name, **predictor(name).explain(body.text)}


@app.post("/api/compare")
def compare(body: TextIn):
    out = []
    for name in available_models():
        p = predictor(name)
        probs = p.probs([body.text])[0]
        i = int(probs.argmax())
        out.append({"model": name, "label": p.labels[i], "confidence": float(probs[i])})
    return {"results": out}


@app.get("/api/results")
def results():
    if not summary():
        raise HTTPException(503, "results/summary.json missing; run python -m fallacylens.analyze")
    return summary()


@app.get("/api/examples")
def get_examples():
    return {"examples": examples()}


if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "unknown API route")
        target = DIST / path
        if path and target.is_file() and DIST in target.resolve().parents:
            return FileResponse(target)
        return FileResponse(DIST / "index.html")
