# FallacyLens — Phase 1 Presentation Guide (for the Mentor Review)

How to present Phase 1 in ~10–12 minutes. Plain language, correct technical
terms. Slots marked `{{FILL: ...}}` get their values from the actual run —
check `results/phase1_results.csv`, `report/PHASE1_REPORT.md`, and the
executed notebook before presenting.

---

## 1. The 60-second pitch

> "FallacyLens is a deep-learning system that detects logical fallacies in
> written arguments — things like ad hominem attacks, false dilemmas, or
> circular reasoning. In Phase 1 I did a systematic literature review of the
> four main fallacy datasets, picked the most scientifically defensible one,
> and built a working CNN text-classification baseline. The results are
> preliminary — Phase 2 will compare CNN, BiLSTM, and Transformer models
> under one consistent protocol."

## 2. Presentation flow (what to say, in order)

1. **The problem** — A logical fallacy is a flaw in reasoning that makes an
   argument invalid or unsound, even when it sounds convincing. Detecting
   them automatically is an NLP problem: the input is raw text, the output is
   a fallacy label. It's hard because fallacies are about *reasoning
   structure*, not keywords — the same sentence can be valid or fallacious
   depending on context.
   - One line on the distinctions: detection (is there a fallacy?) vs
     classification (which one?) vs argument mining (find claims/premises)
     vs fact checking (is the claim true?). FallacyLens does classification.
2. **Why it matters** — online discussions, misinformation analysis, debate
   education, critical-thinking tools. Say: "a decision-support tool, not an
   arbiter of truth."
3. **What previous researchers did** — show the literature review matrix
   (`report/PHASE1_REPORT.md`, Part 2). Point at 3–4 rows: the LOGIC paper,
   MAFALDA, CoCoLoFa, one classical-ML or transformer baseline. Say what
   each contributed and its limitation. Emphasize: "every number in this
   table comes from the original paper."
4. **The research gap** — read the gap paragraph (Part 3). Typical framing:
   fallacy work is fragmented across taxonomies and domains; few controlled
   comparisons of architecture families under one protocol; minority classes
   suffer under imbalance. Then your 3–5 contribution bullets.
5. **Datasets investigated** — show the dataset comparison table (Part 4).
   Walk one row per dataset: size, classes, domain, single/multi-label.
6. **Dataset chosen + why** — "LOGIC —
   2,449 samples (1,849 train / 300 dev / 300 test), 13 fallacy types,
   single-label per released row, public GitHub CSVs, quiz-style examples. Simplest
   defensible choice for a first implementation; the others have incompatible
   taxonomies (MAFALDA 23-class multi-label benchmark, CoCoLoFa 8+none) or are
   a domain-challenge set (LOGICCLIMATE) — Phase 2/3 material."
7. **EDA findings** — show the class-distribution bar chart
   (`figures/`) and text-length plot. Say: "the dataset is
   moderately imbalanced — most frequent class *faulty generalization* 17.25% of
   train, rarest *equivocation* 2.11% — which
   is exactly why accuracy alone is not trustworthy here."
8. **Preprocessing** — "deliberately gentle": whitespace normalization,
   tokenization, vocabulary, padding. Say what you did NOT do: "no stopword
   removal or stemming — words like 'all', 'some', 'therefore' carry
   reasoning signal."
9. **The CNN baseline** — show the printed architecture in the notebook.
   One sentence per layer: Embedding → 1D convolutions (n-gram feature
   detectors) → global max pooling (keep strongest signal) → dropout →
   softmax over 13 classes. Hyperparameters:
   embedding dim 100, 100 filters per kernel size, kernels 3/4/5, dropout 0.5,
   Adam lr 1e-3, batch 32, up to 20 epochs with early stopping (patience 4)
   on dev macro-F1, seed 42.
10. **Why CNN first** — "fast, trains in minutes on CPU, strong classical
    text-classification baseline, and it sets the floor that BiLSTM and
    Transformers must beat in Phase 2."
11. **Results** — show `results/phase1_results.csv` and the confusion matrix.
    Say the numbers out loud: accuracy 0.5284, macro F1 0.4594, weighted
    F1 0.5179 (best dev macro-F1 0.5056, selected epoch 18). Then the key
    sentence: "Macro F1 is the headline metric
    because it weights every fallacy class equally, including rare ones."
    ALWAYS label these as **Phase 1 preliminary baseline results**.
12. **Error analysis** — show 2–3 rows of the error table. Say what pattern
    you found (e.g., confusion between semantically similar fallacies, or
    short ambiguous texts) — only what the table actually shows.
13. **Limitations + Phase 2** — "single dataset, single architecture, no
    tuning, imbalanced classes. Phase 2: BiLSTM + Transformer under the same
    protocol, class-wise comparison; Phase 3: imbalance handling,
    generalization, explainability."

## 3. Live demo script (what to physically show)

1. Open the repo on GitHub (or the project folder) — show the structure.
2. Open `report/PHASE1_REPORT.md` — scroll: matrix → gap → dataset tables.
3. Open `notebooks/phase1_fallacylens.ipynb` (executed, outputs visible):
   - dataset head() + label list
   - class-distribution bar chart
   - text-length histogram
   - printed model architecture
   - training/validation curves
   - classification report + confusion matrix
   - error-analysis table
4. Show `results/phase1_results.csv`.
5. If asked to prove reproducibility: point to the seeds cell (seed = 42),
   `requirements.txt`, and the nbconvert execute command in the README.

## 4. Questions the mentor will probably ask

- **"Why not BERT/Transformers already?"** — Phase 1 scope: establish the
  foundation with a cheap, interpretable baseline; transformers need more
  compute and careful protocol, planned for Phase 2 with a controlled
  comparison.
- **"Why macro F1?"** — fallacy classes are imbalanced; accuracy and
  weighted F1 hide minority-class failure. Macro F1 treats all classes
  equally.
- **"Can you trust these labels?"** — they come from the dataset's published
  annotation process; cite the paper's annotation setup. Different datasets
  define fallacies differently — that's why we did NOT merge taxonomies.
- **"Isn't this just text classification?"** — architecturally yes; the
  research content is in the task definition, taxonomies, evaluation
  protocol, and per-class/error analysis.
- **"How is this different from fake-news detection?"** — fact checking
  verifies truth against external evidence; fallacy detection analyzes the
  reasoning pattern of the argument itself, true premises or not.
- **"What about overfitting?"** — point to the val curves, dropout, early
  stopping, and the held-out test set used exactly once.

## 5. Dos and Don'ts

- DO say "preliminary Phase 1 baseline" every time you quote a number.
- DO say which numbers are from papers vs from your own run.
- DON'T claim state of the art, DON'T claim the system "understands logic",
  DON'T present Phase 2/3 plans as done.
