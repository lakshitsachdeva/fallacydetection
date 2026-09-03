# FallacyLens Phase 1 — Progress Log

Date started: 2026-09-03 (round 1). Task spec: `PHASE1_TASK.md`.

## Status overview
- [x] Repo scaffold inspected; venv verified (torch 2.14.0+cpu, pandas 3.0.5, sklearn 1.9.0, nbformat/nbconvert OK). No GPU. No GitHub credentials (gh not authed, no GITHUB_TOKEN) → will finish with local commits + PUSH_INSTRUCTIONS.md.
- [ ] Literature verification (LOGIC/LOGICCLIMATE, MAFALDA, CoCoLoFa, + more)
- [ ] Dataset download + verification
- [ ] Notebook build + execute
- [ ] Report
- [ ] Final deliverables + PHASE1_COMPLETE.md

## Detailed log
- 2026-09-03: Round 1 started. Nothing pre-existed (no commits, empty data/notebooks/report/results/figures).

## Key facts discovered
- **LOGIC dataset VERIFIED** (2026-09-03): official repo https://github.com/causalNLP/logical-fallacy (real, matches paper). Paper: "Logical Fallacy Detection", Jin, Lalwani, Vaidhya, Shen, Ding, Lyu, Sachan, Mihalcea, Schoelkopf — Findings of EMNLP 2022, https://aclanthology.org/2022.findings-emnlp.532 (DOI 10.18653/v1/2022.findings-emnlp.532), arXiv 2202.13758. Abstract claims structure-aware classifier beats best LM by 5.46% F1 on LOGIC, 4.51% on LOGICCLIMATE.
- LOGIC data files = `edu_train/dev/test.csv` in repo `data/`; LOGICCLIMATE = `climate_*.csv`. Downloaded to `data/raw/` (gitignored).
- LOGIC parsed: train 1849 / dev 300 / test 300 = 2449 total (~2.4k as paper says). Single-label, 13 classes: faulty generalization (319), ad hominem (225), false causality (174), ad populum (158), circular reasoning (134), appeal to emotion (130), fallacy of logic (121), fallacy of relevance (114), intentional (112), false dilemma (110), fallacy of credibility (107), fallacy of extension (106), equivocation (39).
- CSV quirk: many `Unnamed: 0.x` index columns; real columns: `source_article` (original sentence text), `updated_label` (label), `masked_articles` (entity-masked variant). Use `source_article` for our baseline. 11 duplicate texts inside train (dedupe in notebook).
- Env: pandas 3.0.5 / numpy 2.5.2 / sklearn 1.9.0 / torch 2.14.0+cpu. No GPU. No GitHub auth (gh not logged in, no GITHUB_TOKEN) → PUSH_INSTRUCTIONS.md at end.
- Git: initial scaffold commit `812fc84` on main.
