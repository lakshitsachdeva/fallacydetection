# FallacyLens — Project Handoff (Phase 1)

**Date**: 2026-09-04
**Status**: Phase 1 (Literature Review + Initial Implementation) — **complete, verified, committed**

---

## 1. What this project is

FallacyLens is a B.Tech Deep Learning research project: automated detection and
classification of **logical fallacies** in textual arguments. It runs in 3 phases:

- **Phase 1 (DONE)** — literature review, dataset analysis, CNN baseline
- **Phase 2 (planned)** — controlled comparison: improved CNN vs BiLSTM vs Transformer
- **Phase 3 (planned)** — optimization, class-imbalance handling, explainability,
  cross-dataset generalization, final evaluation, report/paper

## 2. How Phase 1 was produced (process record)

1. **Setup session (evening, 2026-09-03)** — from the laptop, the project was
   scaffolded on a remote server (`ubuntu@49.13.90.163`, port 33789):
   - Python venv with torch 2.14 (CPU), pandas, scikit-learn, matplotlib, jupyter
   - A complete Phase 1 task specification (`PHASE1_TASK.md`) encoding the research
     rules: no fabricated papers/results/URLs, primary sources only (ACL Anthology,
     official repos), prior-work numbers vs own-run numbers strictly separated,
     seed = 42 everywhere, Colab-reproducible, CPU-feasible
   - A `VERIFIED_REFERENCES.md` fact sheet, produced by actually fetching each
     source (papers + repos for LOGIC, LOGICCLIMATE, MAFALDA, CoCoLoFa + 4 related
     works), with unverifiable items explicitly marked
   - The run launched in a detached **tmux** session (`fallacylens`) via
     `run_phase1.sh` / `run_inner.sh`, using the launchpad-managed `kmi` agent
     CLI in non-interactive auto-permission mode, with an auto-resume loop
     (up to 8 rounds) that re-prompts from `PROGRESS.md` if the agent exits early
2. **Overnight autonomous run** — the agent verified the literature itself,
   downloaded the LOGIC dataset from its official repo, built and executed the
   notebook end-to-end (`jupyter nbconvert --execute --inplace`), wrote the full
   report, and committed at 6 milestones. Completion was only declared after a
   programmatic deliverables check (`report/PHASE1_COMPLETE.md`).
3. **Morning handoff (2026-09-04)** — artifacts verified from the laptop (results
   CSV matches report and notebook; all 14 code cells executed; guide slots
   filled with real values), full git history pulled back to the laptop, and this
   handoff written.

## 3. Deliverables and where they live

| Deliverable | Path |
|---|---|
| Full Phase 1 report (problem def, lit-review matrix, dataset analysis, research gap, findings, roadmap, mentor walkthrough, IEEE references) | `report/PHASE1_REPORT.md` |
| Executed notebook (30 cells, real outputs) | `notebooks/phase1_fallacylens.ipynb` |
| Results table | `results/phase1_results.csv` |
| Trained CNN checkpoint (1.8 MB, regenerable) | `results/cnn_baseline.pt` |
| Figures | `figures/` (4 PNGs) |
| Presentation guide + demo checklist | `PRESENTATION_GUIDE.md`, `DEMO_CHECKLIST.md` |
| Verified sources fact sheet | `VERIFIED_REFERENCES.md` |
| Original task spec + progress log | `PHASE1_TASK.md`, `PROGRESS.md` |
| Unattended-run scripts (for future phases) | `run_phase1.sh`, `run_inner.sh` |

## 4. Phase 1 results (preliminary baseline — NOT final project results)

Dataset: **LOGIC** (Jin et al., Findings of EMNLP 2022) — 13 fallacy classes,
1,849/300/300 train/dev/test, downloaded from the official repository.

| Model | Accuracy | Precision | Recall | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| CNN baseline (Phase 1) | 0.5284 | 0.4666 | 0.4658 | 0.4594 | 0.5179 |

Settings: seed 42, CPU, early stopping (best dev macro-F1 0.5056 @ epoch 18),
443,813 parameters, ~15 s training. Literature numbers in the report are from
the cited papers only; every own-run number above came from the executed
notebook.

## 5. Reproduce

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute notebooks/phase1_fallacylens.ipynb
```

## 6. Presenting it

Read `PRESENTATION_GUIDE.md` (60-second pitch, 13-step flow, demo script,
likely mentor Q&A). Run `DEMO_CHECKLIST.md` the morning of the review.

## 7. Next steps

- Phase 2 and Phase 3 roadmaps are in `report/PHASE1_REPORT.md` §13.
- The server setup is reusable for Phase 2: `ssh -p 33789 ubuntu@49.13.90.163`,
  `cd ~/fallacydetection`, adapt `PHASE1_TASK.md`, `bash run_phase1.sh`.
- GitHub repo: https://github.com/lakshitsachdeva/fallacydetection
