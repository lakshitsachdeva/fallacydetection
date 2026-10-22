# FallacyLens — Automated Logical Fallacy Detection Using Deep Learning

A B.Tech deep learning project comparing CNN, BiLSTM, and compact pretrained
Transformer classifiers on the LOGIC fallacy dataset. The final experiment also
tests class weighting, a frozen encoder, direct LOGICCLIMATE transfer, and local
token sensitivity. It predicts one of 13 dataset labels; it does not verify
whether an argument is truly fallacious or factually correct.

## Status

**Phase 1 — Literature Review and Initial Implementation**

- Systematic literature review of logical-fallacy detection (LOGIC, LOGICCLIMATE,
  MAFALDA, CoCoLoFa, and related work)
- Dataset investigation and justified selection of the Phase 1 dataset
- Reproducible pipeline: loading → EDA → preprocessing → CNN baseline →
  evaluation → error analysis
- Phase 1 report with research gap, preliminary results, and references

**Phases 2–3 — final experimental comparison**

- Controlled reruns of CNN, BiLSTM, and pretrained BERT-tiny under one protocol
- Class weighting, frozen-encoder, and learning-rate ablations
- Official and overlap-excluded LOGIC test results; direct LOGICCLIMATE transfer
- Per-class metrics, errors, bootstrap interval, token-deletion sensitivity
- Measured outputs in `results/final/`, charts in `figures/final/`, and
  interpretation in `report/FINAL_REPORT.md`

## Layout

| Path | Contents |
| --- | --- |
| `PHASE1_TASK.md` | The complete Phase 1 task specification / research brief |
| `notebooks/` | Executed Phase 1 notebook and final results notebook |
| `report/` | Phase 1 literature review and final project report |
| `figures/` | Phase 1 and final saved plots |
| `results/` | Phase 1 and final measured tables, predictions, protocol, uncertainty |
| `data/raw/` | Official downloaded CSVs (gitignored and fetched by training script) |
| `train_final.py` | Reproducible training and evaluation for the final experiment |
| `analyze_final.py` / `verify_final.py` | Figures, bootstrap and integrity check |
| `demo.py` | Command-line classifier using the dev-selected model |
| `run_phase1.sh` / `run_inner.sh` | Unattended-run launcher (tmux) used on the setup server |

## Reproduce

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python demo.py "Most people agree with me, so I must be right."
.venv/bin/python train_final.py
.venv/bin/python analyze_final.py
.venv/bin/python verify_final.py
```

The first run downloads the official CSVs and the pretrained encoder. The
previous Phase 1 notebook remains independently reproducible with
`.venv/bin/jupyter nbconvert --to notebook --execute notebooks/phase1_fallacylens.ipynb`.
Final settings, source hashes, predictions, and metrics are saved with the
results. The dev-selected CNN checkpoint is included for immediate demo;
other checkpoints and row-level text/prediction files are regenerated locally.
The final notebook displays the aggregate artifacts and runs the pipeline if
they do not exist.
