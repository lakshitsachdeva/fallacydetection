# FallacyLens: Logical Fallacy Classification with Deep Learning

FallacyLens classifies a short argument into one of the 13 fallacy types of the
LOGIC benchmark (Jin et al., Findings of EMNLP 2022). It compares TF-IDF, CNN,
BiLSTM-attention and DistilBERT classifiers under one leakage-controlled
protocol and serves the trained models through a web app with word-level explanations.

| Phase | What it contains |
| --- | --- |
| 1 | Literature review (10 verified papers), dataset audit, CNN baseline: `report/PHASE1_REPORT.md`, `notebooks/phase1_fallacylens.ipynb` |
| 2 | Controlled comparison of all model families, 3 seeds each: `fallacylens/train.py`, `results/` |
| 3 | Class weighting, LOGICCLIMATE transfer, significance tests, occlusion explanations, web app, paper: `fallacylens/analyze.py`, `app/`, `web/`, `paper/` |

The single-seed pilot from the earlier final-experiment PR (`train_final.py`,
`analyze_final.py`, `report/FINAL_REPORT.md`, `results/final/`) is kept for
reference. Its data protocol is reused unchanged by `fallacylens/data.py`.

## Results (strict LOGIC test, mean ± std over seeds 42–44)

The strict test holds 283 sentences and excludes every test text already seen in train or dev.
The full table is in `results/final_results.csv` and the paper.

| Model | Params | Strict macro-F1 | Strict accuracy | LOGICCLIMATE macro-F1 (transfer) |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF + LR | 101K | 0.473 ± 0.000 | 0.523 | 0.127 |
| CNN | 437K | 0.423 ± 0.028 | 0.473 | 0.078 |
| BiLSTM-Att | 584K | 0.358 ± 0.014 | 0.419 | 0.116 |
| DistilBERT | 67M | 0.533 ± 0.008 | 0.585 | 0.133 |
| **DistilBERT (class-weighted)** | 67M | **0.554 ± 0.010** | **0.603** | **0.146** |

Class-weighted DistilBERT is selected on dev macro-F1. Its lead over TF-IDF
(p = 0.002), the CNN and the BiLSTM (p < 0.001) is significant (paired
bootstrap and McNemar, `results/significance.csv`); its gain over unweighted
DistilBERT is not (p = 0.12).
Every model fails on climate news without adaptation.

## Run it

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt   # for a CUDA GPU, install torch from pytorch.org first

python -m fallacylens.train       # all experiments, seeds 42 43 44 (≈20 min on a laptop GPU)
python -m fallacylens.analyze     # tables, figures, significance tests, results/summary.json

cd web && npm install && npm run build && cd ..
python -m uvicorn app.server:app --port 8000
```

Open http://127.0.0.1:8000. For frontend development, run `npm run dev` in
`web/`; Vite proxies `/api` to port 8000.

The app has three tabs:
- **Analyze**: classify an argument with any trained model, see the top-5
  probabilities and word influence (occlusion), and compare all models.
- **Results**: the model comparison, per-class F1 and significance tests.
- **Method**: the pipeline, the three architectures and the project phases.

The UI uses shadcn/ui with [ReUI](https://reui.io) components (Badge, Alert, Stepper, Timeline).

## Vercel deployment

Set the Vercel project's root directory to `web` (Vite preset). Its `api/`
functions provide live inference from the measured TF-IDF + logistic-regression
baseline without PyTorch or a missing neural checkpoint. The Results tab still
shows the saved eight-configuration experiment. The Analyze tab identifies the
live baseline so its prediction is not confused with the selected DistilBERT
result. To regenerate the committed inference artifact after changing the
training protocol, run `python web/scripts/export_baseline.py` from the repo
root; it checks the export against the saved seed-42 predictions.

## Layout

| Path | Contents |
| --- | --- |
| `fallacylens/` | data protocol, models, training, analysis, inference + explanations |
| `app/server.py` | FastAPI backend (`/api/analyze`, `/api/compare`, `/api/results`, ...) |
| `web/` | React + Vite + Tailwind frontend |
| `results/` | `final_results.csv`, `per_class_f1_strict.csv`, `significance.csv`, `summary.json`, per-run JSON in `runs/` |
| `figures/final/` | comparison, per-class heatmap, confusion matrix, class-weighting effect, dev curves |
| `paper/` | IEEE-format paper (`fallacylens_ieee.tex`, compiled PDF) |
| `docs/SPEAKER_NOTES.md` | four-person presentation script |
| `docs/MASTER_GUIDE.md` | everything about the project in one file: lit review, protocol, all results, file map, viva Q&A |

## Limits

The label set has no "not a fallacy" class, so every input gets a fallacy
label. LOGIC is small and quiz-style. Explanations show model sensitivity,
not reasoning. Predictions are hypotheses, not judgements of truth.
