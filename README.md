# FallacyLens — Automated Logical Fallacy Detection Using Deep Learning

A B.Tech Deep Learning research project. The system identifies logical fallacies
in textual arguments and, across three phases, will compare CNN, BiLSTM, and
Transformer-based text classifiers.

## Status

**Phase 1 — Literature Review and Initial Implementation** (this branch/state)

- Systematic literature review of logical-fallacy detection (LOGIC, LOGICCLIMATE,
  MAFALDA, CoCoLoFa, and related work)
- Dataset investigation and justified selection of the Phase 1 dataset
- Reproducible pipeline: loading → EDA → preprocessing → CNN baseline →
  evaluation → error analysis
- Phase 1 report with research gap, preliminary results, and references

Phase 2 (model development/comparison) and Phase 3 (improvement, explainability,
final evaluation) are planned but **not** part of this phase.

## Layout

| Path | Contents |
| --- | --- |
| `PHASE1_TASK.md` | The complete Phase 1 task specification / research brief |
| `notebooks/` | Executable Phase 1 notebook (runs top-to-bottom on Colab or CPU) |
| `report/` | Phase 1 report: literature review, gap, findings, references |
| `figures/` | Saved plots (class distribution, training curves, confusion matrix) |
| `results/` | Metrics tables (CSV/Markdown) produced by actually running the notebook |
| `data/` | Dataset download scripts / small processed artifacts |
| `run_phase1.sh` / `run_inner.sh` | Unattended-run launcher (tmux) used on the setup server |

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute notebooks/phase1_fallacylens.ipynb
```

All reported results come from executing the notebook; random seeds and
experimental settings are recorded inside it.
