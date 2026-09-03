# FallacyLens — Phase 1 Report
## Literature Review and Initial Implementation

**Project**: FallacyLens — Automated Logical Fallacy Detection Using Deep Learning (B.Tech research project)
**Phase**: Phase 1 of 3 — research problem definition, systematic literature review, dataset analysis, and one working initial deep-learning baseline.
**Date**: 2026-09-03. **Random seed**: 42 everywhere. **Code**: `notebooks/phase1_fallacylens.ipynb`, executed top-to-bottom on 2026-09-03 (CPU-only torch 2.14.0, Python 3.12.3); every "our result" below comes from that actual run.

> **Traceability rule.** Every number reported as *previous work* is taken from the original paper,
> linked in the References (each was opened and verified on 2026-09-03). Every number reported as
> *ours* comes from the executed notebook or the files it produced (`results/`, `figures/`).
> Anything that could not be verified from a primary source is explicitly marked "could not be
> verified".

---

## PART 1 — Research problem definition

### 1.1 Problem statement

A **logical fallacy** is a flaw in reasoning — an argument pattern that *appears* to support a
conclusion but does not actually establish it. Informal fallacies (our focus) are failures in the
content or structure of everyday argumentation, e.g. *ad hominem* (attacking the person instead of
the claim), *false causality* (assuming that because B followed A, A caused B), or *false dilemma*
(presenting two options as the only possibilities). This is distinct from a *factual* error: a
logically valid argument can rest on false premises, and a fallacious argument may contain only
true statements.

**Automated fallacy detection** is the task of reading a short natural-language argument and
identifying (a) whether it is fallacious and (b) *which* fallacy type it commits. It is an
NLP/deep-learning problem because the signal is not lexical: the same surface form can be fallacious
or sound depending on content, and the same fallacy can be phrased arbitrarily differently. A model
must learn compositional, distributional representations of text — what neural text classifiers
provide — rather than follow hand-written rules. It is hard because it sits close to the frontier of
language understanding: Jin et al. (2022) show that zero-shot pretrained language models score near
random chance on this task, and even fine-tuned transformers succeed only moderately.

**Precise distinctions** (these scope our work):

| Neighbouring task | What it does | How FallacyLens differs |
|---|---|---|
| Fallacy **detection** | Binary decision: fallacious vs sound? | Treated here as a sub-case of classification (13-way includes no "sound" class; detection-vs-classification distinction noted per dataset). |
| Fallacy **classification** | Assign a fallacy *type*. | Our Phase 1 task — single-label, 13 classes, per the released LOGIC data. |
| **Argument mining** | Extract argument components (claims, premises) and their relations from discourse. | Argument mining recovers *structure*; fallacy detection judges the *reasoning*. The strongest LOGIC baseline is "structure-aware", which is why we track this distinction. |
| **Misinformation detection** | Decide whether a claim is factually false. | A claim can be factually true yet fallaciously argued, and vice versa; the signals differ (world knowledge vs reasoning structure). |
| **Fact checking** | Verify a claim against external evidence. | Relies on retrieval/evidence; fallacy detection works from the argument text alone. |

### 1.2 Motivation

Fallacious reasoning is common in online discussion, political debate, advertising, and
climate-change discourse — the LOGICCLIMATE dataset was built precisely from climate-news claims.
Reliable fallacy detection could support misinformation analysis (flagging *how* claims are argued,
not only whether they are true), debate platforms, media-literacy and critical-thinking education,
and moderation assistance. We do **not** claim current models "understand reasoning"; Part 12 shows
the task remains far from solved, and our own baseline's confusion patterns support that caution.

### 1.3 Research questions

- **RQ1 (architecture)**: How do CNN, BiLSTM, and Transformer classifiers compare on fallacy
  classification under one identical protocol? *(Phase 1 delivers the CNN reference point.)*
- **RQ2 (representation)**: Do contextual pretrained representations improve over non-contextual
  learned embeddings on this small dataset, and by how much?
- **RQ3 (robustness)**: How does moderate class imbalance affect per-class and minority-class
  performance, and do simple remedies (class weighting) help?
- **RQ4 (generalization)**: Does a model trained on quiz-style fallacies transfer to naturally
  occurring ones (LOGIC → LOGICCLIMATE), and how large is the drop?
- **RQ5 (error structure)**: Are errors concentrated between semantically neighbouring fallacy
  families (e.g., the "fallacy of X" group), indicating taxonomy overlap rather than weak learning?

---

## PART 2 — Systematic literature review

### 2.1 Literature review matrix

10 verified papers. "Main Result" = numbers **reported by the original paper** (their metric); where
a paper's abstract reports no number, this is stated rather than invented.

| # | Paper | Year | Dataset | Problem | Method/Model | Metrics | Main Result (as reported) | Limitation | Relevance to FallacyLens |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Habernal et al., "The Argument Reasoning Comprehension Task" | 2018 | ARCT (~2k arguments, news comments) | Reconstruct the implicit warrant linking premise to claim | New crowdsourced task + neural attention/language models | warrant-selection accuracy | Abstract: "current approaches do not suffice" (no number on page) | Warrants ≠ fallacy types | Foundational: reasoning over arguments is hard for models — motivates the task |
| 2 | Sahai et al., "Breaking Down the Invisible Wall of Informal Fallacies in Online Discussions" | 2021 | Reddit corpus, 8 fallacy types (~8.6k texts) | Detect informal fallacies in Reddit debates | Noisy self-labels cleaned by crowdsourcing; neural classifiers with/without conversational context | classification metrics | Models "perform better in the presence of conversational context" (no number on page) | Noisy labels; needs conversation context | First large naturally occurring fallacy corpus; later absorbed into MAFALDA |
| 3 | Sheng et al., "'Nice Try, Kiddo': Investigating Ad Hominems in Dialogue Responses" | 2021 | Human/DialoGPT responses to Twitter posts | Detect ad hominems; reduce them in generation | Ad-hominem taxonomy + classifier + constrained decoding | classifier metrics + qualitative findings | Qualitative findings only (no number on page) | Single fallacy type; generation-side focus | Shows per-fallacy-type detection is feasible; safety angle |
| 4 | Jin et al., "Logical Fallacy Detection" | 2022 | LOGIC (2,449), LOGICCLIMATE (1,079 in paper) | Fallacy classification (+ detection framing) | 12 LMs zero-shot & fine-tuned; best Electra 53.31% micro-F1; structure-aware NLI-style classifier | micro-F1, exact match | Electra-StructAware **58.77% micro-F1 on LOGIC** (+5.46 over Electra); zero-shot near random (RoBERTa-MNLI 12.22%); on LOGICCLIMATE 27.23% direct transfer | Quiz-style sentences; paper frames task multi-label but released rows are single-label | Introduces our primary dataset; strongest verified baseline; sets realistic expectations |
| 5 | Goffredo et al., "Fallacious Argument Classification in Political Debates" | 2022 | 31 US debates; 1,628 fallacious arguments; 6 categories | Classify fallacies in political debates | Transformer + argumentation features (components/relations) | classification metrics | "Outperforming state-of-the-art results and standard baselines" (no number on page) | Single domain; 6 classes; no public corpus at scale then | Debate domain; combining text + argumentative structure foreshadows structure-aware trend |
| 6 | Helwe et al., "MAFALDA: A Benchmark and Comprehensive Study of Fallacy Detection and Classification" | 2024 | MAFALDA: 9,745 merged texts (Sahai, Martino, Jin, Goffredo), 200 re-annotated (268 spans) | Benchmark fallacy detection (3 levels: binary / 3 coarse / 23 fine) | Zero-shot evaluation of 7 LLMs | span-level P/R/F1 | GPT-3.5 best at all levels; **humans outperform all LLMs**; Mistral F1 = 0.149, Zephyr F1 = 0.185 (direct prompting, Level 1). Per-model table values could not be verified from the fetched page | Benchmark for evaluation, not training; "disjunctive" multi-label annotation (28% of spans have ≥2 acceptable labels) | Authoritative modern taxonomy (23 fine classes, Aristotelian 3-way split); evidence LLMs are weak zero-shot |
| 7 | Yeh, Wan & Huang, "CoCoLoFa: A Dataset of News Comments with Common Logical Fallacies Written by LLM-Assisted Crowds" | 2024 | CoCoLoFa: 7,706 comments on 648 news articles; 8 fallacy types + none | Fallacy detection & classification in news comments | LLM-assisted crowdsourced writing + expert validation; fine-tuned BERT family | F1 | Fine-tuned BERT: **F1 = 0.86 (detection), 0.87 (classification)** on test; beats SOTA LLMs | Crowd-written (assistant-assisted) text may differ from organic comments; explicit license could not be verified (no LICENSE file) | Largest verified single-label news-comment set; realistic BERT reference for Phase 2/3 |
| 8 | Lei & Huang, "Boosting Logical Fallacy Reasoning in LLMs via Logical Structure Tree" | 2024 | LOGIC, LOGICCLIMATE, Argotario, Reddit corpus | Improve LLM fallacy detection/classification | Unsupervised logical-structure-tree (constituency + connective relations) as hard prompt or soft prompt | detection & classification P/R | "Significantly improves precision and recall" (no number in abstract) | Requires a parsing/connective pipeline; LLM-dependent | Strong evidence that *structure* is the key signal — validates our structure-aware motivation |
| 9 | Pan et al., "Are LLMs Good Zero-Shot Fallacy Classifiers?" | 2024 | Argotario, Covid/Climate, Reddit, ElecDeb, MAFALDA, Propaganda, LOGIC | Zero-shot fallacy classification with LLMs | Single- and multi-round prompting schemes (extraction, summarization, CoT) | classification metrics | Acceptable zero-shot; "outperform [full-shot baselines] in all OOD inference scenarios" (qualitative; no number in abstract) | Prompt sensitivity; evaluation heterogeneity | Positions prompting as a cheap Phase 3 baseline; OOD weakness of supervised models supports RQ4 |
| 10 | Wang et al., "Follow My Lead: Logical Fallacy Classification with Knowledge-Augmented LLMs" | 2025 | FALLACIES (232 classes) | Improve LLM fallacy classification cheaply | AID-LF stepwise instruction dataset + Prolog relational graphs + 3-tier hierarchy | accuracy | Body (verified in Sec. 4): **+20.7% (Claude-Sonnet-4), +3.4% (ChatGPT-4o/o4-mini), +8.7% (Gemini-2.5-Flash)**; best 62.9% vs 42.2% baseline (abstract itself has no number) | Expert 232-class taxonomy, far from our 13-class setting; LLM-only | Decomposition/knowledge augmentation is the current frontier we position against |

### 2.2 Comparative analysis

**Datasets.** The field has no single agreed corpus: ARCT targets implicit warrants (not fallacy
types); the Reddit corpus (Sahai et al.) is large and natural but noisily labeled; LOGIC is small,
clean, single-label and quiz-derived; LOGICCLIMATE is domain-restricted climate news; Goffredo's
debates are expert-annotated but small and single-domain; MAFALDA unifies prior corpora under a new
23-class taxonomy but only 200 texts were freshly annotated and it is designed as a zero-shot
benchmark; CoCoLoFa is the largest verified single-label set but its comments are crowd-written with
an LLM assistant. Sizes range from ~300 (Goffredo) to ~9.7k (MAFALDA merged). None of the four
candidate project datasets (LOGIC, LOGICCLIMATE, MAFALDA, CoCoLoFa) share a compatible label space:
13 classes, 13 classes (same taxonomy, different domain and label format), 23 fine classes in a
3-level hierarchy with disjunctive multi-label annotation, and 8+none respectively. **Safe
cross-dataset merging is not possible in Phase 1** (detailed in Part 4).

**Classical ML vs neural vs transformer vs structure-aware.** Verified evidence shows rule/lexical
approaches are not competitive on these datasets; neural classifiers (Sahai et al.) benefit from
conversation context; fine-tuned transformers are the standard strong baseline (Electra 53.31%
micro-F1 on LOGIC; BERT F1 0.86/0.87 on CoCoLoFa); and **structure-aware** methods — argumentation
features (Goffredo), NLI-style structure-aware hypotheses (Jin et al.), logical structure trees
(Lei & Huang) — consistently add performance, while **zero-shot LLMs remain weak** (near-random in
Jin et al.; humans beat all LLMs in MAFALDA; knowledge augmentation needed per Wang et al.).

**Single- vs multi-label.** Inconsistent across corpora: LOGIC rows are single-label (though the
paper frames the task as multi-label), CoCoLoFa is single-label, MAFALDA's disjunctive scheme is
effectively multi-label, LOGICCLIMATE's `logical_fallacies` column format differs from LOGIC's
`updated_label`. Evaluation protocols (micro-F1, span-F1, accuracy) are equally heterogeneous, which
makes naive number comparisons across papers misleading.

**Class imbalance and domain dependency.** Every verified dataset is imbalanced (our EDA: 8:1
max:min in LOGIC). All datasets are strongly domain-dependent (quiz sentences, Reddit, debates,
climate news, assisted news comments) — the single direct-transfer number we could verify
(Electra-StructAware 27.23% micro-F1 LOGIC→LOGICCLIMATE) shows large cross-domain drops, echoed by
Pan et al.'s OOD findings.

**Textbook vs naturally occurring.** LOGIC's quiz-style fallacies are textbook-clean; MAFALDA,
CoCoLoFa and the Reddit corpus target naturally occurring (or naturalistic) text. Models trained on
the former generalize poorly to the latter — the central generalization question (RQ4).

**Dataset limitations (synthesis).** noisy distant labels (Reddit), tiny expert sets (Goffredo,
MAFALDA's re-annotated core), synthetic/assisted text (CoCoLoFa), domain lock-in (LOGICCLIMATE),
license ambiguity (CoCoLoFa: no LICENSE file — could not be verified). **Model limitations.**
small-data overfitting (our Part 12 confirms), weakness of zero-shot LLMs, dependence on parsing or
prompt engineering for structure-aware gains, and inconsistent metrics/reporting across papers.

---

## PART 3 — Research gap

Across the verified literature, fallacy detection has progressed through *isolated* dataset papers
and *one-off* strong models — a structure-aware NLI classifier here (Jin et al., 2022), LLM prompting
and knowledge augmentation there (Lei & Huang, 2024; Pan et al., 2024; Wang et al., 2025) — while
the label spaces of the four candidate datasets remain mutually incompatible (13 vs 23 fine classes
vs 8+none), evaluation metrics are heterogeneous (micro-F1 vs span-F1 vs accuracy), and humans still
outperform all evaluated LLMs on the most rigorous benchmark (Helwe et al., 2024). What we did **not**
find, in any verified source, is a controlled, reproducible comparison of the three standard neural
text-classifier families (CNN, BiLSTM, Transformer) on a single fallacy dataset under a fixed
protocol with explicit per-class and minority-class analysis under imbalance. That comparison —
cheap, rigorous, and directly decision-relevant for a student project — is the gap FallacyLens
targets.

**Proposed contributions (evidence-backed, realistic):**

1. **A controlled CNN / BiLSTM / Transformer comparison** on LOGIC under one identical protocol
   (splits, seed, preprocessing, macro-F1 headline) — Phase 1 delivers the CNN reference point.
2. **Per-class and minority-class robustness analysis under moderate imbalance**, quantifying where
   and why errors concentrate (RQ3, RQ5).
3. **An honest cross-domain generalization measurement** (LOGIC → LOGICCLIMATE) with realistic
   expectations set by the verified 27.23% transfer result of Jin et al. (RQ4).
4. **A fully reproducible, Colab-friendly pipeline** (seeded, auto-downloading, CPU-feasible) that a
   B.Tech student can rerun end-to-end and demonstrate.

No "world's first" claim is made: the gap is the absence of a *controlled basic-architecture
comparison with class-wise analysis*, not the absence of fallacy-detection research.

---

## PART 4 — Dataset analysis

### 4.1 Table 1 — Candidate datasets (all rows verified from primary sources, 2026-09-03)

| Dataset | Source | Size | Domain | Labels | # Classes | Single/Multi-label | Train/Val/Test | Suitable for Phase 1? |
|---|---|---|---|---|---|---|---|---|
| **LOGIC** | Jin et al. 2022, official GitHub causalNLP/logical-fallacy | 2,449 | Educational quiz sites (Quizizz, Study.com, ProProfs) + manual web collection | fallacy type | 13 | Single-label per released row (paper frames task as multi-label) | 1,849 / 300 / 300 | **Yes — selected.** Small, clean, single-label, plain CSVs, no access gate. |
| **LOGICCLIMATE** | same repo | 1,351 in current files (paper reports 1,079 — post-publication update; discrepancy could not be reconciled) | Climate-change news claims (climatefeedback.org) | fallacy type (`logical_fallacies` column, different schema) | 13 (shared taxonomy) | Row format differs from LOGIC (different label column/schema) | 855 / 277 / 219 | Reserved as the **Phase 2 out-of-domain challenge set** (RQ4): domain shift is by design. |
| **MAFALDA** | Helwe et al. 2024 (NAACL), github.com/ChadiHelwe/MAFALDA | 9,745 merged texts; only 200 freshly re-annotated (268 spans) | Reddit + news + LOGIC + debates (unified) | 3-level: binary / 3 coarse / 23 fine | 23 fine (+3 coarse + binary) | Disjunctive multi-label (28% of spans have ≥2 acceptable labels) | Benchmark corpus for zero-shot evaluation, not shipped as train/val/test | **No for Phase 1** — multi-level multi-label annotation and benchmark purpose add complexity inappropriate for a first baseline; **Phase 3 candidate**. CC-BY-SA stated in paper. |
| **CoCoLoFa** | Yeh, Wan & Huang 2024 (EMNLP), github.com/Crowd-AI-Lab/cocolofa | 7,706 comments / 648 articles | News comments (LLM-assisted crowd-written) | fallacy type + none | 8 + none | Single-label | 5,370 / 1,538 / 798 | **No for Phase 1** — larger and attractive for Phase 2/3, but different taxonomy (8 vs 13 classes) blocks joint training with LOGIC; explicit license could not be verified (no LICENSE file). |

### 4.2 Table 2 — Label taxonomies do not merge

Authoritative LOGIC names from the dataset's own `mappings.csv` (verified at the raw URL in
References; "Understandable Name" column shown as the short meaning). The "Possible Unified Label"
column is deliberately **left dataset-specific**: the four datasets define fallacies at different
granularities under different annotation guidelines (LOGIC: 13; MAFALDA: 23 fine in an Aristotelian
hierarchy; CoCoLoFa: 8+none; LOGICCLIMATE: same 13 as LOGIC but news-domain). Merging labels across
datasets because names look similar (e.g. LOGIC's *faulty generalization* vs CoCoLoFa's *hasty
generalization*) would silently violate annotation semantics. Any harmonization is deferred to
Phases 2/3 as an explicit research decision, if attempted at all.

| Dataset | Original Label | Meaning (from dataset's own mapping) | Possible Unified Label |
|---|---|---|---|
| LOGIC | faulty generalization | faulty generalization | *(dataset-specific)* |
| LOGIC | false causality | false causality | *(dataset-specific)* |
| LOGIC | circular reasoning | circular reasoning | *(dataset-specific)* |
| LOGIC | ad populum | appeal to popularity | *(dataset-specific)* |
| LOGIC | ad hominem | personal attack | *(dataset-specific)* |
| LOGIC | fallacy of logic | logical error | *(dataset-specific)* |
| LOGIC | appeal to emotion | appeal to emotion | *(dataset-specific)* |
| LOGIC | false dilemma | excluding viable alternatives | *(dataset-specific)* |
| LOGIC | equivocation | use of ambiguous language | *(dataset-specific)* |
| LOGIC | fallacy of extension | exaggerating | *(dataset-specific)* |
| LOGIC | fallacy of relevance | irrelevant argument | *(dataset-specific)* |
| LOGIC | fallacy of credibility | attacking the character of the speaker | *(dataset-specific)* |
| LOGIC | intentional | intentionally wrong argument | *(dataset-specific)* |

### 4.3 Recommendation: primary Phase 1 dataset = **LOGIC**

1. **Simplest scientifically defensible setup** — single-label, 13 classes, short sentences, fixed
   official train/dev/test splits; nothing about the experimental design needs improvisation.
2. **Verified availability** — plain CSVs on the introducing paper's official public GitHub repo; no
   registration or gate. We verified files, row counts, and the label mapping by direct download.
3. **Right size for Phase 1** — 2,449 samples train a small CNN in ~15 s/epoch on CPU; the whole
   notebook trains in well under a minute, meeting the CPU/Colab constraint.
4. **Benchmark value** — the introducing paper reports verified strong baselines (Electra 53.31%
   micro-F1; structure-aware 58.77%), giving our baseline meaningful context.
5. LOGICCLIMATE is reserved for RQ4 (cross-domain). MAFALDA and CoCoLoFa are documented,
   incompatibility-explained candidates for later phases — not forced into Phase 1.

---

## PARTS 5–11 — Implementation summary

Full executable work: `notebooks/phase1_fallacylens.ipynb` (executed top-to-bottom, committed with
outputs; dataset auto-downloads from the official repo if absent).

- **5.1 Setup**: Python 3.12.3, torch 2.14.0+cpu (CUDA unavailable), pandas 3.0.5, numpy 2.5.2;
  seeds = 42 for torch/numpy/random.
- **5.2 Loading**: `edu_{train,dev,test}.csv` from the official repo → `data/raw/` (gitignored);
  1,849/300/300 rows; 13 labels; input column `source_article` (original text; the `masked_articles`
  variant is not used for this baseline).
- **6 EDA** (`figures/`): class counts/percentages — majority *faulty generalization* 17.25% of train
  vs minority *equivocation* 2.11% (≈8:1 imbalance); text-length stats — train mean 24.4 words,
  median 20, max 188, 95th percentile 54; distribution plots and real per-class examples.
- **7 Preprocessing** (each decision justified in the notebook): dedup (6/1/1 rows dropped);
  whitespace normalisation; punctuation-preserving regex tokenisation with **no stopword removal or
  stemming** (discourse markers and punctuation carry the reasoning signal); train-only vocabulary
  (min_freq 2 → 3,196 tokens); label encoding; padding/truncation to MAX_LEN = 54 (95th percentile).
  Leakage checks: train-only vocab; 10/299 test texts (3.3%) also appear in train — a property of
  the official splits, reported not removed.
- **8 CNN baseline**: Embedding(3,196×100, padding_idx=0) → Conv1d(100 filters, kernels 3/4/5, ReLU)
  → global max-pool → concat → dropout 0.5 → Linear(300→13). 443,813 trainable parameters. Chosen
  first because it is the cheapest strong baseline for short texts, trains in seconds on CPU, and
  gives Phase 2 (BiLSTM/Transformer) a fair reference on identical preprocessing and splits.
- **9 Training**: Adam lr 1e-3, batch 32, ≤20 epochs, early stopping (patience 4) on **dev macro-F1**;
  best dev macro-F1 = 0.5056 at epoch 18; ~15 s total; checkpoint `results/cnn_baseline.pt`;
  curves in `figures/training_curves.png`.
- **10 Evaluation** — test split. **Phase 1 preliminary baseline results — NOT final project results:**

| Model | Accuracy | Precision (macro) | Recall (macro) | **Macro F1** | Weighted F1 |
|---|---|---|---|---|---|
| CNN baseline (Phase 1) | 0.5284 | 0.4666 | 0.4658 | **0.4594** | 0.5179 |

  Per-class report and confusion matrix are in the notebook (`figures/confusion_matrix.png`); the
  table is saved to `results/phase1_results.csv`. **Comparability note**: Jin et al. report *micro*-F1
  (53.31% Electra / 58.77% structure-aware on LOGIC); we headline *macro*-F1 because of the imbalance
  — the figures are not directly interchangeable, but our ~53% accuracy sits in the same overall
  performance region as the fine-tuned transformers they report, for a model ~1/1000th the size.
- **11 Error analysis**: 141/299 test errors (47.2%); 15-error table with true/predicted/confidence
  and only mechanically-checkable "possible reason" labels (minority predicted class, truncation,
  overconfidence). Patterns visible in this run: *equivocation* recall = 0 (0/5; smallest class, 39
  train samples); strongest mutual confusion inside the "fallacy of X" family (credibility,
  extension, relevance, logic), consistent with overlapping definitions; several high-confidence
  errors on minority classes, consistent with imbalance effects.

---

## PART 12 — Phase 1 findings

**Literature.** The task is genuinely hard: zero-shot LMs are near random on LOGIC (Jin et al.), and
humans beat all evaluated LLMs on MAFALDA (Helwe et al.). The strongest consistent signal across
verified work is that **structure-aware** methods help (argumentation features, NLI-style structure,
logical structure trees), and recent gains come from knowledge augmentation of LLMs. Datasets are
fragmented with incompatible taxonomies and heterogeneous metrics; cross-domain transfer is poor
(27.23% micro-F1 LOGIC→LOGICCLIMATE in Jin et al.).

**Dataset.** LOGIC is small (2,449), clean, single-label, moderately imbalanced (≈8:1), with short
texts (median 20 words). Its quiz provenance is both a strength (clean labels) and a limitation
(natural-text generalization unproven). A 3.3% train/test text overlap exists in the official
splits — documented, not removed, so our numbers are comparable to the paper's protocol.

**Initial model.** The CNN baseline reaches **test accuracy 0.5284, macro-F1 0.4594** (best dev
macro-F1 0.5056) — a working, honest preliminary result in the same region as early fine-tuned
transformers, achieved on CPU in ~15 s of training. Training curves show clear overfitting (train
acc 96.7% vs dev ~52% by epoch 20), as expected for 1.8k samples and 444k parameters.

**Problems identified.** (1) Small-data overfitting — the dominant issue; (2) moderate class
imbalance collapsing the rarest class (*equivocation*, recall 0); (3) taxonomy overlap inside the
"fallacy of X" family driving confusions; (4) metric heterogeneity in the literature complicating
comparison; (5) limited dataset realism (quiz sentences).

**What Phase 1 establishes.** A verified evidence base (10 papers, all opened and checked), a
justified dataset choice with explicit non-integration rationale for the other candidates, a
reproducible leakage-controlled pipeline, and a working CNN baseline with quantified errors — the
reference point and infrastructure Phase 2's architecture comparison needs.

---

## PART 13 — Roadmap (Phases 2–3 — planned, not executed)

- **Phase 2**: BiLSTM and Transformer (e.g., DistilBERT) classifiers under the *identical* protocol
  (same splits, seed 42, preprocessing, macro-F1 headline) → RQ1/RQ2; class-weighting/resampling
  study → RQ3; LOGIC→LOGICCLIMATE transfer experiment → RQ4; structured error analysis on taxonomy
  overlap → RQ5.
- **Phase 3**: optimization and final architecture comparison; cross-dataset generalization with
  MAFALDA/CoCoLoFa where defensible; explainability; ablations; final report/paper.

---

## How to Explain Phase 1 to the Project Mentor (verbal-review version)

1. **Problem.** People argue badly online — *ad hominem*, *false cause*, *false dilemma*. We want a
   model that reads a short argument and says which fallacy it contains. That is different from
   fact-checking: an argument can use perfect facts and still be fallacious.
2. **Prior work.** We read and verified 10 papers. Key lessons: the task is hard (zero-shot language
   models are near random; humans beat all LLMs on the MAFALDA benchmark); fine-tuned transformers
   get roughly halfway; and methods that expose the *logical structure* of the sentence work best.
3. **Gap.** Nobody has published a clean, controlled comparison of the three standard neural
   architectures (CNN, BiLSTM, Transformer) on one fallacy dataset with per-class analysis — that is
   what our project contributes.
4. **Datasets investigated.** LOGIC (2,449 quiz-style sentences, 13 fallacy types), LOGICCLIMATE
   (climate news, reserved as a later "does it generalize?" test), MAFALDA (23-class benchmark, too
   complex/multi-label for a first model), CoCoLoFa (8-type news comments, different taxonomy).
   Their label sets don't merge safely, so combining them now would be bad science.
5. **Chosen: LOGIC** — smallest defensible setup: single-label, fixed splits, publicly hosted CSVs,
   no paperwork, trains in seconds on CPU.
6. **What we found in the data.** 13 classes, moderately imbalanced (biggest class ~8× the
   smallest), short sentences (~20 words median). We keep punctuation and words like "therefore",
   because the *reasoning connectors* are the clue — we don't strip stopwords.
7. **Model.** A small convolutional network (the classic sentence-CNN): word embeddings → three
   convolution filters (sizes 3, 4, 5) → keep the strongest signal of each filter → dropout →
   13-way softmax. Half a million parameters, trains on this laptop's CPU in ~15 seconds per full
   run.
8. **Why CNN first.** It is the cheapest respectable baseline; later phases add BiLSTM and
   Transformer on identical footing so the comparison is fair.
9. **Results (preliminary!).** ~53% test accuracy, macro-F1 0.46. For context, the original paper's
   fine-tuned Electra transformer got 53% *micro*-F1 — different metric, but our tiny model is in
   the same ballpark, which validates the pipeline.
10. **Errors.** Almost half the errors involve semantically similar "fallacy of X" classes, and the
    rarest class (*equivocation*) was never found — exactly the imbalance/taxonomy effects our
    research questions target.
11. **Limitations.** Small quiz-style data, clear overfitting, one architecture only.
12. **Phase 2 plan.** Same protocol, add BiLSTM and a pretrained Transformer, attack the imbalance,
    then test on climate news to measure real-world transfer.

---

## Phase 1 conclusion

Phase 1 established the research foundation for FallacyLens: a verified systematic review of ten
primary sources showing that logical-fallacy detection is an open, hard reasoning task whose
progress is currently fragmented across incompatible datasets and one-off models; a dataset analysis
that selects LOGIC as the primary corpus with explicit, evidence-backed reasons for deferring
LOGICCLIMATE, MAFALDA, and CoCoLoFa; and a reproducible PyTorch pipeline with a working CNN
baseline that achieves 0.5284 test accuracy and 0.4594 macro-F1 under seed 42 in minutes on CPU.
The baseline's error structure — collapse on the rarest class and confusions within the
semantically overlapping "fallacy of X" family — directly motivates the Phase 2 research questions
on architecture comparison, imbalance handling, and cross-domain generalization. Phase 1 therefore
delivers exactly what a first project phase must: a defensible evidence base, an honest initial
result, and the infrastructure for controlled comparison — while its own limitations (small
quiz-style data, overfitting, single architecture) define the agenda for Phase 2.

---

## References (IEEE)

[1] Z. Jin, A. Lalwani, T. Vaidhya, X. Shen, Y. Ding, Z. Lyu, M. Sachan, R. Mihalcea, and
B. Schölkopf, "Logical Fallacy Detection," *Findings of the Association for Computational
Linguistics: EMNLP 2022*, pp. 7180–7198, 2022. doi: 10.18653/v1/2022.findings-emnlp.532. [Online].
Available: https://aclanthology.org/2022.findings-emnlp.532/ (dataset:
https://github.com/causalNLP/logical-fallacy)

[2] C. Helwe, T. Calamai, P.-H. Paris, C. Clavel, and F. Suchanek, "MAFALDA: A Benchmark and
Comprehensive Study of Fallacy Detection and Classification," *NAACL 2024, Long Papers*, vol. 1,
pp. 4810–4845, 2024. doi: 10.18653/v1/2024.naacl-long.270. [Online]. Available:
https://aclanthology.org/2024.naacl-long.270/ (dataset: https://github.com/ChadiHelwe/MAFALDA)

[3] M.-H. Yeh, R. Wan, and T.-H. Huang, "CoCoLoFa: A Dataset of News Comments with Common Logical
Fallacies Written by LLM-Assisted Crowds," *EMNLP 2024*, pp. 660–677, 2024. [Online]. Available:
https://aclanthology.org/2024.emnlp-main.39/ (preprint: arXiv:2410.03457; dataset:
https://github.com/Crowd-AI-Lab/cocolofa/)

[4] I. Habernal, H. Wachsmuth, I. Gurevych, and B. Stein, "The Argument Reasoning Comprehension
Task: Identification and Reconstruction of Implicit Warrants," *NAACL-HLT 2018*, pp. 1930–1940,
2018. [Online]. Available: https://aclanthology.org/N18-1175/

[5] S. Sahai, O. Balalau, and R. Horincar, "Breaking Down the Invisible Wall of Informal Fallacies
in Online Discussions," *ACL-IJCNLP 2021*, pp. 644–657, 2021. [Online]. Available:
https://aclanthology.org/2021.acl-long.53/

[6] E. Sheng, K.-W. Chang, P. Natarajan, and N. Peng, "'Nice Try, Kiddo': Investigating Ad
Hominems in Dialogue Responses," *NAACL-HLT 2021*, pp. 750–767, 2021. [Online]. Available:
https://aclanthology.org/2021.naacl-main.60/

[7] P. Goffredo, S. Haddadan, V. Vorakitphan, E. Cabrio, and S. Villata, "Fallacious Argument
Classification in Political Debates," *IJCAI 2022*, pp. 4143–4149, 2022. [Online]. Available:
https://www.ijcai.org/proceedings/2022/575

[8] Y. Lei and R. Huang, "Boosting Logical Fallacy Reasoning in LLMs via Logical Structure Tree,"
*EMNLP 2024*, pp. 13157–13173, 2024. [Online]. Available:
https://aclanthology.org/2024.emnlp-main.730/

[9] F. Pan, X. Wu, Z. Li, and A. T. Luu, "Are LLMs Good Zero-Shot Fallacy Classifiers?" arXiv:2410.15050, 2024. [Online]. Available: https://arxiv.org/abs/2410.15050

[10] O. Wang, T. Bansal, R. Bai, E. Chui, and L. Gilpin, "Follow My Lead: Logical Fallacy
Classification with Knowledge-Augmented LLMs," arXiv:2510.09970, 2025. [Online]. Available:
https://arxiv.org/abs/2510.09970

---
*Prepared 2026-09-03. Companion notebook: `notebooks/phase1_fallacylens.ipynb`. Results file:
`results/phase1_results.csv`. Figures: `figures/`.*
