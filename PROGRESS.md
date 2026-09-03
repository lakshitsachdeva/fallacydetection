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
- **MAFALDA VERIFIED** (Helwe, Calamai, Paris, Clavel, Suchanek; NAACL 2024 long, pp. 4810–4845, https://aclanthology.org/2024.naacl-long.270/, repo github.com/ChadiHelwe/MAFALDA, CC-BY-SA): 9,745 merged texts (Sahai/Martino/Jin/Goffredo), only 200 re-annotated (268 spans); 3-level taxonomy (binary / 3 Aristotelian coarse / 23 fine); disjunctive multi-label; zero-shot benchmark — GPT-3.5 best, humans beat all LLMs; Mistral F1=0.149, Zephyr 0.185 @Level 1. Per-model Table 1 values not machine-readable — "could not be verified".
- **CoCoLoFa VERIFIED** (Yeh, Wan, Huang; **EMNLP 2024** main pp. 660–677 — the "NAACL 2025" belief was WRONG; https://aclanthology.org/2024.emnlp-main.39/, arXiv 2410.03457, repo github.com/Crowd-AI-Lab/cocolofa): 7,706 LLM-assisted crowd-written comments on 648 articles; 8 fallacies + none; single-label; splits 5,370/1,538/798; BERT F1 0.86 detection / 0.87 classification. No LICENSE file — license terms could not be verified.
- **7 more papers verified** for the matrix: Habernal et al. 2018 (ARCT, NAACL N18-1175); Sahai et al. 2021 (Reddit corpus, ACL 2021.acl-long.53); Sheng et al. 2021 (ad hominem, NAACL 2021.naacl-main.60); Goffredo et al. 2022 (debates, IJCAI 2022/575); Lei & Huang 2024 (LogicTree, EMNLP 2024.emnlp-main.730); Pan et al. 2024 (zero-shot LLM, arXiv 2410.15050); Wang et al. 2025 (AID-LF, arXiv 2510.09970).
- LOGICCLIMATE current repo files verified locally: 855/277/219 = 1,351 (paper says 1,079 — unreconciled post-publication update, documented); schema differs (`logical_fallacies` label column).
- **Notebook built & executed** (30 cells, ~34 s wall / 15 s training): best dev macro-F1 0.5056 @epoch 18; **test acc 0.5284, macro-F1 0.4594, weighted F1 0.5179**; equivocation recall 0 (0/5); 141/299 errors; vocab 3,196; MAX_LEN 54; params 443,813; dedup 6/1/1; train/test overlap 10/299 (3.3%). Figures: class_distribution, text_length_distribution, training_curves, confusion_matrix. results/phase1_results.csv written. Commit cbc419c.
- **PHASE1_REPORT.md written** (all parts; 10-paper matrix; gap; mentor section; IEEE refs). CoCoLoFa venue corrected to EMNLP 2024 in report.
- **Supporting files** (created by research subagents during this run, verified accurate, kept + integrated): `VERIFIED_REFERENCES.md` (fact sheet), `PRESENTATION_GUIDE.md` (all `{{FILL}}` slots filled with real run values per task §9), `DEMO_CHECKLIST.md` (unticked — for the student), `PHASE1_TASK.md` (spec amended with §9 presentation-file requirement).
- **Single-label vs multi-label decision**: released LOGIC rows each carry exactly one label (verified — no multi-label entries in any split). Baseline treats LOGIC as single-label per released data; paper's multi-label framing documented in report Part 4. This is the documented "justify restricting to single-label" option (task §4 amendment).
- **FINAL**: deliverables 1–6 verified programmatically (notebook 30 cells / 0 errors / all outputs; results CSV matches report; all report sections present). Wrote `report/PHASE1_COMPLETE.md` (last) and `PUSH_INSTRUCTIONS.md` (no GitHub credentials on machine). Final commit below.

## Git: initial scaffold commit `812fc84` on main.
