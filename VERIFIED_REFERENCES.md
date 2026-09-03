# Verified Reference Fact Sheet — Logical Fallacy Detection

Verified 2026-09-03 by fetching each source. Treat as starting points:
re-open each URL when you cite it. "UNVERIFIED" means exactly that — do not
guess a value.

## 1. LOGIC (Jin et al. 2022)
- "Logical Fallacy Detection"; Zhijing Jin, Abhinav Lalwani, et al.;
  **Findings of EMNLP 2022**, pp. 7180–7198.
- https://aclanthology.org/2022.findings-emnlp.532/ (DOI 10.18653/v1/2022.findings-emnlp.532)
- Official repo: https://github.com/causalNLP/logical-fallacy — CSVs
  `data/edu_train.csv`, `edu_dev.csv`, `edu_test.csv`.
- **13 fallacy types**: faulty generalization, false causality, circular
  claim/reasoning, ad populum, ad hominem, appeal to emotion, fallacy of
  credibility, fallacy of logic, fallacy of relevance, fallacy of extension,
  equivocation, false dilemma, intentional.
- Size **2,449**; splits from repo CSVs: **train 1,849 / dev 300 / test 300**.
- **Multi-label** — paper §4.1: multi-label classification with class
  imbalance; **micro F1 is the paper's main metric**.
- Domain: educational/quiz examples (Quizziz, study.com, ProProfs) plus ~600
  manually collected.
- License: no LICENSE file → UNVERIFIED.
- Reported baselines (micro-F1, LOGIC): Electra 53.31% F1; proposed
  Electra-StructAware 58.77% F1 (exact match 47.67%); zero-shot
  RoBERTa-MNLI 12.22% F1.

## 2. LOGICCLIMATE (same paper, Jin et al. 2022)
- Same paper/repo (climate CSVs). Sentences from climate-change news articles
  (climatefeedback.org), span-annotated.
- Size: paper §2.2 says **1,079** samples; contributions section says 1,109 —
  internal inconsistency, quote with caution. Repo climate CSVs sum to 1,351.
- Same 13-type scheme, multi-label.
- Reported (direct transfer from LOGIC): Electra-StructAware 27.23% F1
  (+4.51 over Electra ≈ 22.72%).
- License: UNVERIFIED.

## 3. MAFALDA (Helwe et al. 2024)
- "MAFALDA: A Benchmark and Comprehensive Study of Fallacy Detection and
  Classification"; Helwe, Calamai, Paris, Clavel, Suchanek;
  **NAACL 2024 (Long Papers)**, pp. 4810–4845. arXiv: 2311.09761.
- https://aclanthology.org/2024.naacl-long.270/
- Official repo: https://github.com/ChadiHelwe/MAFALDA
- Taxonomy 3 levels: L0 binary fallacy/no-fallacy; L1 three Aristotelian
  categories; L2 **23 fine-grained fallacy types**.
- Merged corpus **9,745 texts**; 200 manually re-annotated texts with 268
  spans (203 fallacious). Sources include Reddit (Sahai et al. 2021), news
  (Da San Martino et al. 2019), Jin et al. 2022, US debates (Goffredo 2022).
- **Multi-label by design** (disjunctive annotation; 28% of spans have ≥2
  gold labels). Span-level detection+classification.
- **No train/dev/test split — zero-shot benchmark, not intended for
  training.** Reported: "GPT 3.5 outperforms all other models at all levels";
  exact table numbers UNVERIFIED (image-embedded).
- License: UNVERIFIED.

## 4. CoCoLoFa (Yeh et al. 2024)
- "CoCoLoFa: A Dataset of News Comments with Common Logical Fallacies Written
  by LLM-Assisted Crowds"; Yeh, Wan, Huang; **EMNLP 2024 main**, pp. 660–677.
  arXiv: 2410.03457.
- https://aclanthology.org/2024.emnlp-main.39/ (DOI 10.18653/v1/2024.emnlp-main.39)
- Official repo: https://github.com/Crowd-AI-Lab/cocolofa
- **8 fallacy types + "none"**: appeal to authority, appeal to majority,
  appeal to nature, appeal to tradition, appeal to worse problems, false
  dilemma, hasty generalization, slippery slope.
- **7,706 comments on 648 news articles**; fallacious 4,576 / non-fallacious
  3,130. Splits: train 5,370 / dev 1,538 / test 798 comments.
- **Single-label per comment** (detection binary; classification 8-way).
- License: UNVERIFIED.
- Reported (abstract): fine-tuned BERT-based models F1=0.86 detection,
  F1=0.87 classification on its test set.

## Other verified prior work worth citing
- Habernal, Wachsmuth, Gurevych, Stein, "Before Name-Calling: Dynamics and
  Triggers of Ad Hominem Fallacies in Web Argumentation," NAACL 2018,
  pp. 386–396, https://aclanthology.org/N18-1036/ — early Reddit ad-hominem
  corpus, cited by Jin et al.
- Stab & Gurevych, "Recognizing Insufficiently Supported Arguments in
  Argumentative Essays," EACL 2017, pp. 980–990,
  https://aclanthology.org/E17-1092/ — precursor argument-sufficiency task.
- Da San Martino et al., "Fine-Grained Analysis of Propaganda in News
  Articles," EMNLP-IJCNLP 2019, pp. 5636–5646,
  https://aclanthology.org/D19-1565/ — 18 propaganda techniques overlapping
  fallacies; span metric reused by MAFALDA.
- Sourati, Ilievski, Sandlin, Mermoud, "Case-Based Reasoning with Language
  Models for Classification of Logical Fallacies," IJCAI 2023,
  pp. 5188–5196, https://arxiv.org/abs/2301.11879 — main post-LOGIC method.

## Known traps
- LOGIC/LOGICCLIMATE are **multi-label**; CoCoLoFa is **single-label**.
- MAFALDA has **no training split** — evaluation-only benchmark.
- LOGICCLIMATE size is inconsistently reported (1,079 vs 1,109).
- None of the three GitHub repos declares a license.
