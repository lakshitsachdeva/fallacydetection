# Phase 1 Complete — FallacyLens

**Date**: 2026-09-03. All Phase 1 deliverables exist, were verified against the task spec
(`PHASE1_TASK.md`), and are committed on `main`.

## What was produced

| Deliverable | Where | Status |
|---|---|---|
| Full Phase 1 report (parts 1–4, 12–13, literature matrix, dataset analysis, gap, results, mentor walkthrough, conclusion, IEEE refs) | `report/PHASE1_REPORT.md` | ✅ |
| Executed notebook (setup → loading → EDA → preprocessing → CNN → training → evaluation → error analysis, 30 cells, 0 errors, all outputs present) | `notebooks/phase1_fallacylens.ipynb` | ✅ |
| Results table | `results/phase1_results.csv` | ✅ |
| Figures (class distribution, text length, training curves, confusion matrix) | `figures/` (4 PNGs) | ✅ |
| Progress log | `PROGRESS.md` | ✅ |
| Mentor presentation guide (all `{{FILL}}` slots filled from the real run) + demo checklist + verified-references fact sheet | `PRESENTATION_GUIDE.md`, `DEMO_CHECKLIST.md`, `VERIFIED_REFERENCES.md` | ✅ |
| Git milestones (6 commits) | `git log` | ✅ |

## Key numbers (from the actual executed run, seed 42, CPU)

- Dataset: **LOGIC** (Jin et al., Findings of EMNLP 2022, official repo) — 1,849/300/300, 13 classes.
- CNN baseline: **test accuracy 0.5284, macro-F1 0.4594, weighted F1 0.5179**; best dev macro-F1
  0.5056 (epoch 18, early stopping); ~15 s training on CPU; 443,813 parameters.
- Labeled everywhere as **Phase 1 preliminary baseline — NOT final project results**.

## Verification performed

- Every literature number traced to a fetched primary source (ACL Anthology / IJCAI / arXiv / official
  repos); unverifiable items marked "could not be verified" (e.g., CoCoLoFa license, MAFALDA per-model
  table, LOGICCLIMATE 1,079-vs-1,351 size discrepancy).
- Notebook executed via `.venv/bin/jupyter nbconvert --to notebook --execute --inplace`; committed
  outputs match `results/phase1_results.csv` and the report exactly.
- Full deliverables checklist (spec §6, items 1–6) confirmed programmatically before this file was written.

## Remaining (not part of Phase 1 deliverables)

- Push to GitHub: no credentials on this machine — see `PUSH_INSTRUCTIONS.md`.
