# FallacyLens — Master Project Guide

This is the single reference for the whole project: what it is, how it was built,
every result, where every file lives, and answers to the questions an examiner is
likely to ask. Every number here comes from `results/` (final study) or from the
cited papers (literature). Nothing is estimated.

---

## Contents

1. [The project in one minute](#1-the-project-in-one-minute)
2. [Key concepts you must be able to explain](#2-key-concepts)
3. [The 13 fallacy classes](#3-the-13-fallacy-classes)
4. [Literature review](#4-literature-review)
5. [Datasets and why we chose LOGIC](#5-datasets-and-why-we-chose-logic)
6. [Data protocol (leakage control)](#6-data-protocol-leakage-control)
7. [Preprocessing](#7-preprocessing)
8. [Models](#8-models)
9. [Training, selection and hardware](#9-training-selection-and-hardware)
10. [Evaluation and statistics](#10-evaluation-and-statistics)
11. [Results — everything](#11-results--everything)
12. [Error analysis and explanations](#12-error-analysis-and-explanations)
13. [How the project evolved (Phase 1 → pilot → final)](#13-how-the-project-evolved)
14. [The web application](#14-the-web-application)
15. [Where everything is (repository map)](#15-where-everything-is)
16. [How to reproduce everything](#16-how-to-reproduce-everything)
17. [Limitations, ethics, future work](#17-limitations-ethics-future-work)
18. [Viva / examiner questions and answers](#18-viva--examiner-questions-and-answers)
19. [Numbers cheat sheet](#19-numbers-cheat-sheet)

---

## 1. The project in one minute

**FallacyLens** reads a short argument and names the logical fallacy it commits,
choosing one of the 13 classes of the LOGIC benchmark (Jin et al., Findings of
EMNLP 2022).

What we did:
- Cleaned LOGIC with a **leakage-controlled protocol**. We removed conflicting labels and duplicates, removed train/dev
  overlap, and scored models on a **strict test of 283 sentences** that no model had seen in train or dev.
- Compared **four model families** under exactly the same protocol: TF-IDF + logistic regression, CNN,
  BiLSTM with attention, and DistilBERT.
- Ran each family **with and without class-weighted loss**, over **3 random seeds** (42, 43, 44).
  That makes 8 configurations and 24 training runs.
- Tested **significance** with a paired bootstrap and McNemar's test.
- Tested **domain transfer** to climate-news fallacies (LOGICCLIMATE) without any retraining.
- Built a **web app** (FastAPI + React + shadcn/ReUI) that runs the real models and shows which words drove each prediction.
- Wrote an **IEEE-format paper** (PDF and Word).

Headline result (strict test, mean ± std over 3 seeds):

| Model | Macro-F1 |
|---|---|
| **DistilBERT, class-weighted (selected on dev)** | **0.554 ± 0.010** |
| DistilBERT | 0.533 ± 0.008 |
| TF-IDF + LR | 0.473 |
| CNN | 0.423 ± 0.028 |
| BiLSTM-Att | 0.358 ± 0.014 |

**Takeaways:**
- **Pretraining is what separates the models.** With only 1,801 training sentences, the networks trained from
  scratch lose even to a TF-IDF linear model.
- **Class weighting is a small, non-significant extra.** It does not fix rare classes.
- **Every model collapses on climate news**, to macro-F1 of 0.146 at best.

**60-second pitch (say this out loud):**
> "An argument can use only true facts and still be wrong, like *everyone uses this phone, so it's the best*.
> That flaw is a logical fallacy. FallacyLens is a deep-learning system that names the fallacy in a sentence.
> We found duplicated and leaked sentences in the standard benchmark, cleaned it, and compared TF-IDF, CNN, BiLSTM
> and DistilBERT fairly over three seeds. The pretrained DistilBERT wins clearly, at 0.55 macro-F1, and the win is
> statistically significant. Simple class weighting doesn't fix rare fallacies, and no model transfers to climate
> news. Our web app shows the prediction and the words that drove it."

---

## 2. Key concepts

| Term | Plain explanation |
|---|---|
| **Logical fallacy** | A flaw in reasoning. The argument appears to support its conclusion but doesn't. It can contain only true statements. |
| **Fallacy detection vs classification** | Detection asks "is there a fallacy?" (yes/no). Classification asks "which fallacy?". We do **classification**, with 13 classes. |
| **vs fact-checking** | Fact-checking checks whether a claim is true against evidence. We check whether the *reasoning* holds, from the text alone. |
| **vs argument mining** | Argument mining finds claims, premises and their relations. We judge the reasoning pattern. |
| **Single-label** | Each LOGIC row has exactly one label. The original paper discusses a multi-label framing, but the released rows are single-label. |
| **Macro-F1** | Compute F1 for each class, then take the plain average. Every class counts equally, so it exposes failure on rare classes. **Our headline metric.** |
| **Accuracy** | The fraction of predictions that are correct. For single-label tasks it equals micro-F1. |
| **Weighted F1** | Per-class F1 averaged with weights equal to class size. Dominated by large classes. |
| **Class weighting** | Errors on rare classes cost more in the loss: w_c = N / (K · n_c). |
| **Data leakage** | Test (or dev) information leaking into training, for example the same sentence appearing in both. It inflates scores. |
| **Strict test** | The official test set minus every sentence that also appears in train or dev. 283 of 300 rows remain. |
| **Seed** | The random-number starting point. Different seeds give different weight initialisations and batch orders, so we average 3. |
| **Pretraining** | DistilBERT first learned general English from a large corpus. We then *fine-tune* it on LOGIC. |
| **Occlusion explanation** | Delete one word at a time and see how much the predicted class's probability drops. A big drop means an important word. |
| **Paired bootstrap** | Resample the 283 test rows with replacement 2,000 times and recompute the macro-F1 difference between two models. This gives a 95% confidence interval. |
| **McNemar's test** | Counts the sentences only model A got right versus only model B got right, and tests whether that split is unbalanced. |
| **Domain transfer** | Applying a model trained on one kind of text (quiz sentences) to another (climate news) without retraining. |

---

## 3. The 13 fallacy classes

Definitions come from the dataset's own `mappings.csv`. *n* is the number of training sentences after cleaning.

| Class (dataset label) | Readable name | Meaning | n (train) |
|---|---|---|---|
| faulty generalization | faulty generalization | A conclusion about all or many cases is drawn from one or a few. | 314 |
| ad hominem | personal attack | Attacks the person instead of their argument. | 219 |
| false causality | false causality | Claims A caused B without evidence. Often "after A, therefore because of A". | 167 |
| ad populum | appeal to popularity | Something must be true or good because many people believe or do it. | 150 |
| circular reasoning | circular reasoning | The conclusion is assumed in the premise. | 131 |
| appeal to emotion | appeal to emotion | Manipulates emotions instead of giving reasons. | 127 |
| fallacy of logic | logical error | A general error in the logical structure. | 119 |
| fallacy of relevance | irrelevant argument | Premises unrelated to the subject (red herring). | 111 |
| intentional | intentionally wrong argument | A deliberate, sometimes subconscious, wrong move to support an argument. | 110 |
| false dilemma | excluding viable alternatives | Only two options are presented when more exist. | 109 |
| fallacy of credibility | attacking the character of the speaker / appeal to authority | Rests on the speaker's credibility rather than the argument. | 107 |
| fallacy of extension | exaggerating (straw man) | Attacks an exaggerated version of the opponent's position. | 100 |
| equivocation | use of ambiguous language | A key word changes meaning midway through the argument. | 37 |

The most frequent class has 314 examples and the rarest has 37, an imbalance of about 8.5 : 1.

**Examples to use in the viva:**
- Ad populum: "Everyone in my class uses this phone, so it must be the best one."
- False dilemma: "Either we ban all cars or the city will choke on smog." (our app predicts false dilemma, 99.5%)
- Ad hominem: "You can't trust his climate argument; he failed high-school science."
- Circular reasoning: "This book is true because the book says it is true."

---

## 4. Literature review

Ten papers were verified against their primary sources in Phase 1 (`report/PHASE1_REPORT.md`, `VERIFIED_REFERENCES.md`).
Numbers are as **reported by each paper**.

| # | Paper | Venue | Dataset | Method | Key reported result | Limitation | Relevance to us |
|---|---|---|---|---|---|---|---|
| 1 | Habernal et al., *Argument Reasoning Comprehension Task* | NAACL 2018 | ARCT, ~2k news-comment arguments | Neural attention / LMs choose the implicit warrant | "Current approaches do not suffice" | Warrants, not fallacy types | Shows reasoning over arguments is hard |
| 2 | Habernal et al., *Before Name-Calling* | NAACL 2018 | Reddit ad hominem | Analysis + classifiers | Dynamics and triggers of ad hominem | One fallacy type | Early fallacy data |
| 3 | Sahai et al., *Breaking Down the Invisible Wall…* | ACL 2021 | Reddit, 8 fallacy types | Crowd-cleaned labels; neural classifiers ± context | Conversational context helps | Noisy labels | First large natural corpus |
| 4 | Sheng et al., *"Nice Try, Kiddo"* | NAACL 2021 | Dialogue responses | Ad hominem classifier + constrained decoding | Qualitative findings | One type; generation focus | Per-type detection is feasible |
| 5 | **Jin et al., *Logical Fallacy Detection*** | Findings EMNLP 2022 | **LOGIC (2,449), LOGICCLIMATE** | 12 LMs fine-tuned; structure-aware classifier | Electra 53.31% micro-F1; **structure-aware 58.77%**; zero-shot near random (12.22%); climate transfer 27.23% | Quiz-style text | **Our dataset and baseline reference** |
| 6 | Goffredo et al., *Fallacious Argument Classification in Political Debates* | IJCAI 2022 | 31 US debates, 6 classes | Transformer + argument features | Beats baselines | Single domain | Structure helps |
| 7 | Helwe et al., **MAFALDA** | NAACL 2024 | 9,745 merged texts, 200 re-annotated | Zero-shot LLM benchmark; 3-level taxonomy (23 fine classes) | GPT-3.5 best; **humans beat all LLMs** | Evaluation-only, multi-label | Modern taxonomy; LLMs weak |
| 8 | Yeh, Wan & Huang, **CoCoLoFa** | EMNLP 2024 | 7,706 news comments, 8 types + none | LLM-assisted crowd writing; BERT | BERT F1 0.86 detection / 0.87 classification | Synthetic-ish text | Largest single-label set; future work |
| 9 | Lei & Huang, *Logical Structure Tree* | EMNLP 2024 | LOGIC, LOGICCLIMATE, Argotario, Reddit | Structure-tree prompts for LLMs | Better precision and recall | Needs parsing pipeline | Structure is key |
| 10 | Pan et al., *Are LLMs Good Zero-Shot Fallacy Classifiers?* | arXiv 2024 | 7 datasets | Zero-shot prompting schemes | Competitive out-of-domain | Prompt sensitivity | Prompting as future baseline |
| 11 | Wang et al., *Follow My Lead* (AID-LF) | arXiv 2025 | FALLACIES, 232 classes | Knowledge-augmented LLMs | +20.7% for Claude-Sonnet-4, etc. | 232-class taxonomy | Current frontier |
| 12 | Sourati et al., *Case-Based Reasoning…* | IJCAI 2023 | LOGIC etc. | Retrieval of similar cases + LMs | Improves classification | Retrieval cost | Post-LOGIC method |

Model and method papers we build on: Kim 2014 (sentence CNN), Hochreiter & Schmidhuber 1997 (LSTM),
Bahdanau et al. 2015 (attention), Vaswani et al. 2017 (Transformer), Devlin et al. 2019 (BERT), Sanh et al. 2019
(DistilBERT), Kingma & Ba 2015 (Adam), Loshchilov & Hutter 2019 (AdamW), Efron & Tibshirani 1993 (bootstrap),
McNemar 1947, Dietterich 1998 (statistical tests for classifiers), Zeiler & Fergus 2014 (occlusion).
All 23 are listed in IEEE format in `paper/refs.bib` and in the paper.

**Synthesis (what the literature says):**
- The task is genuinely hard. Zero-shot models are near random, and humans beat LLMs.
- Structure-aware methods help. These include argument features, masked logical form and structure trees.
- Datasets are fragmented. Taxonomies are incompatible (13 vs 23 vs 8+none), and metrics differ (micro-F1, span-F1, accuracy).
- Cross-domain transfer is poor.

**Research gap:** no controlled, reproducible comparison of the standard neural families (CNN, BiLSTM, Transformer)
on one fallacy dataset, under a fixed protocol with leakage control, multiple seeds, per-class analysis and
significance tests. **That is our contribution.** We make no "world's first" claim.

**Research questions:**
- RQ1: How do TF-IDF, CNN, BiLSTM and DistilBERT compare under one protocol?
- RQ2: Do pretrained contextual representations beat embeddings learned from scratch on small data?
- RQ3: Does class weighting help rare classes?
- RQ4: Do models transfer from LOGIC to LOGICCLIMATE?
- RQ5: Where do errors concentrate?

---

## 5. Datasets and why we chose LOGIC

| Dataset | Size | Domain | Classes | Labels | Splits | Used? |
|---|---|---|---|---|---|---|
| **LOGIC** | 2,449 | Educational quiz sites | 13 | Single-label per row | 1,849 / 300 / 300 | **Yes, primary** |
| **LOGICCLIMATE** | 1,351 in the current repo (the paper says 1,079) | Climate-change news | Same 13 | Single label column | 855 / 277 / 219 | **Yes, transfer test only** (test file, 211 usable rows) |
| MAFALDA | 9,745 merged, 200 re-annotated | Mixed | 23 fine + 3 coarse + binary | Multi-label | No training split | No. It is an evaluation-only benchmark with an incompatible taxonomy. |
| CoCoLoFa | 7,706 | News comments | 8 + none | Single | 5,370 / 1,538 / 798 | No. Different taxonomy, no license file. Future work. |

**Why LOGIC:**
- It is clean and single-label, with fixed official splits.
- It is public at a pinned commit, with no gate.
- It is small enough for multi-seed runs on a laptop GPU.
- Its paper gives strong reference baselines.

We **never merged taxonomies**. Labels with similar names can have different definitions across datasets.

LOGICCLIMATE test label counts: intentional 45, faulty generalization 28, appeal to emotion 26, fallacy of relevance 22,
fallacy of credibility 20, fallacy of logic 17, ad hominem 16, false causality 16, false dilemma 6, fallacy of
extension 5, ad populum 5, equivocation 4, circular reasoning 1. This distribution is very different from LOGIC's.

---

## 6. Data protocol (leakage control)

Code: `fallacylens/data.py` → `load_protocol()`. Recorded in `results/protocol.json`. The protocol was introduced in
PR #1 and reused unchanged.

1. **Pinned download.** CSVs come from `causalNLP/logical-fallacy` at commit `2a3d0970…`, and each file's SHA-256 is
   checked. If the data ever changes, the code refuses to run.
2. **Normalise.** Collapse whitespace and casefold to build a comparison key.
3. **Conflicting labels.** 5 training texts appear with two different labels. The text cannot decide between them,
   so all their copies are dropped: 1,849 → 1,839.
4. **Deduplicate train:** 1,839 → 1,814.
5. **Remove train ∩ dev:** 1,814 → **1,801 training sentences**. This stops early stopping from rewarding memorisation.
6. **Strict test.** Of the 300 official test sentences, **17** also appear in train or dev. Removing them leaves
   **283 strict-test sentences**. The official 300-row test is also reported for reference.
7. **Climate test.** 219 rows, minus 8 with empty text → **211**.
8. **Test labels are never used** for training, early stopping, model selection or tuning.

Text length (train, words): mean 24.1, median 20, min 3, max 188, 95th percentile 54.

---

## 7. Preprocessing

- Whitespace normalisation only. **No stopword removal, no stemming, punctuation kept.** Words like *all*, *everyone*,
  *therefore*, *because*, *either/or* are the clues. Our occlusion probe confirmed the models rely on exactly these words.
- Tokeniser for CNN, BiLSTM and TF-IDF: regex `[A-Za-z0-9]+(?:'[a-z]+)?|[^\w\s]`, lowercased.
- Vocabulary is built from **training text only**, with minimum frequency 2. `<PAD>` = 0 and `<UNK>` = 1.
- Input length: CNN and BiLSTM are truncated or padded to **54 tokens** (the 95th percentile). DistilBERT uses its own
  WordPiece tokenizer, up to **128** subword tokens.
- Labels are sorted alphabetically and encoded as 0–12.

---

## 8. Models

Code: `fallacylens/models.py` and `fallacylens/train.py`.

| Model | Architecture | Trainable params |
|---|---|---|
| **TF-IDF + LR** | Unigram + bigram TF-IDF (min_df 2, sublinear tf) → multinomial logistic regression, C = 10 | 101K weights |
| **CNN** (Kim 2014) | Embedding 100-d (learned from scratch) → Conv1d, 100 filters each of width 3/4/5, ReLU → global max-pool → concat (300) → dropout 0.5 → Linear(13) | 436,613 |
| **BiLSTM-Att** | Embedding 100-d → dropout → BiLSTM, 128 units per direction → additive attention (Linear 256→128, tanh, Linear→1, softmax over time, padding masked) → weighted sum → dropout 0.5 → Linear(13) | 584,286 |
| **DistilBERT** | `distilbert-base-uncased`: 6 Transformer layers, 768 hidden, 12 heads. Pre-classifier + classifier head on the [CLS] token. Fine-tuned end to end. | 66,963,469 |

**Class-weighted variant ("cw"):** the same model with `CrossEntropyLoss(weight = N / (13 · n_c))`. For TF-IDF it uses
`class_weight="balanced"`, which is the same formula.

**Why these models:**
- TF-IDF is the classic non-neural baseline.
- The CNN is the cheapest strong neural sentence classifier, and the Phase 1 baseline.
- The BiLSTM tests whether modelling word order helps.
- DistilBERT tests pretraining, at a size that fits a laptop GPU with 3 seeds.

---

## 9. Training, selection and hardware

| Setting | CNN / BiLSTM | DistilBERT |
|---|---|---|
| Optimiser | Adam, lr 1e-3 | AdamW, lr 5e-5, weight decay 0.01 |
| Batch size | 32 | 16 |
| Epochs | ≤ 20, early stopping with patience 4 | 8 (best epoch kept) |
| Schedule | constant | 10% linear warm-up, then linear decay |
| Extras | — | gradient clipping 1.0, mixed precision (fp16) |
| Model selection | epoch with best **dev macro-F1** | same |

- **Seeds:** 42, 43 and 44 for every configuration, covering Python, NumPy and PyTorch (CPU and CUDA).
- **Hardware:** NVIDIA RTX 3050 laptop GPU (6 GB), PyTorch 2.14 (CUDA 13), transformers 5.20, Windows 11.
- **Time per run:** TF-IDF about 2 s, CNN about 10 s, BiLSTM about 10 s, DistilBERT about 2 minutes.
  All 24 runs took about 20 minutes.
- **Best epochs** (seeds 42/43/44):
  - CNN: 17, 12, 10
  - BiLSTM: 14, 20, 13
  - DistilBERT: 8, 6, 6
  - DistilBERT-cw: 8, 8, 6
- **Selected model:** the configuration with the highest **mean dev macro-F1**, which is DistilBERT (cw) at 0.632.
  The test set was not involved in the choice.

---

## 10. Evaluation and statistics

- **Splits reported:** dev, official test (300), **strict test (283, primary)**, climate test (211).
- **Metrics:** accuracy, macro precision, macro recall, **macro-F1**, weighted F1, per-class F1.
- **Aggregation:** mean ± sample standard deviation over 3 seeds.
- **Significance:** computed on seed-42 predictions, on the strict test, for the selected model against every other configuration.
  - **Paired bootstrap:** 2,000 resamples of the 283 rows. Each time we compute macro-F1(A) − macro-F1(B); the
    2.5th and 97.5th percentiles give the 95% CI. If the CI excludes 0, the difference is significant.
  - **Exact McNemar test:** counts the sentences only A got right (b) and only B got right (c), and runs a
    two-sided binomial test on b against b + c with p = 0.5.

Code: `fallacylens/analyze.py`. Outputs go to `results/significance.csv`.

---

## 11. Results — everything

### 11.1 Full table (mean ± std over seeds 42–44)

| Model | Params | Dev macro-F1 | Official test acc / macro-F1 | **Strict acc** | **Strict macro-F1** | Strict weighted F1 | Climate acc | **Climate macro-F1** |
|---|---|---|---|---|---|---|---|---|
| TF-IDF + LR | 101K | 0.489 | 0.537 / 0.492 | 0.523 | 0.473 ± 0.000 | 0.509 | 0.171 | 0.127 |
| TF-IDF + LR (cw) | 101K | 0.498 | 0.537 / 0.492 | 0.523 | 0.475 ± 0.000 | 0.518 | 0.161 | 0.128 |
| CNN | 437K | 0.444 ± 0.026 | 0.489 / 0.441 | 0.473 | 0.423 ± 0.028 | 0.464 | 0.117 | 0.078 |
| CNN (cw) | 437K | 0.449 ± 0.035 | 0.482 / 0.445 | 0.464 | 0.426 ± 0.012 | 0.459 | 0.115 | 0.090 |
| BiLSTM-Att | 584K | 0.391 ± 0.022 | 0.432 / 0.369 | 0.419 | 0.358 ± 0.014 | 0.410 | 0.126 | 0.116 |
| BiLSTM-Att (cw) | 584K | 0.357 ± 0.046 | 0.353 / 0.339 | 0.346 | 0.333 ± 0.091 | 0.344 | 0.114 | 0.083 |
| DistilBERT | 67.0M | 0.621 ± 0.010 | 0.601 / 0.550 | 0.585 | 0.533 ± 0.008 | 0.583 | 0.161 | 0.133 |
| **DistilBERT (cw)** ✔ selected | 67.0M | **0.632 ± 0.017** | **0.618 / 0.570** | **0.603** | **0.554 ± 0.010** | **0.598** | **0.174** | **0.146** |

TF-IDF has std 0.000 because logistic regression on fixed features is deterministic; the seed doesn't change it.
The selected model's strict macro precision is 0.592 and its macro recall is 0.546.

Files: `results/final_results.csv`. Figure: `figures/final/model_comparison.png`.

### 11.2 Is the winner really better? (strict test, seed 42)

| DistilBERT (cw) vs | Δ macro-F1 | 95% bootstrap CI | only-A correct / only-B correct | McNemar p | Significant? |
|---|---|---|---|---|---|
| TF-IDF + LR | +0.087 | [+0.034, +0.146] | 41 / 17 | 0.002 | **Yes** |
| TF-IDF + LR (cw) | +0.085 | [+0.030, +0.141] | 44 / 20 | 0.004 | **Yes** |
| CNN | +0.143 | [+0.080, +0.205] | 54 / 17 | < 0.001 | **Yes** |
| CNN (cw) | +0.124 | [+0.067, +0.182] | 57 / 17 | < 0.001 | **Yes** |
| BiLSTM-Att | +0.201 | [+0.131, +0.270] | 67 / 16 | < 0.001 | **Yes** |
| BiLSTM-Att (cw) | +0.163 | [+0.094, +0.227] | 75 / 18 | < 0.001 | **Yes** |
| DistilBERT (plain) | +0.036 | [+0.002, +0.074] | 18 / 9 | 0.122 | **No** (McNemar) |

**How to say it:** "Pretraining gives a real, significant gain over every non-pretrained model. Class weighting on
top of DistilBERT is a small refinement that we can't prove is real."

### 11.3 Per-class F1 (strict test, mean of seeds)

| Class | n | TF-IDF | CNN | CNN cw | BiLSTM | DistilBERT | **DistilBERT cw** |
|---|---|---|---|---|---|---|---|
| faulty generalization | 314 | 0.54 | 0.52 | 0.50 | 0.50 | 0.66 | 0.64 |
| ad hominem | 219 | 0.67 | 0.56 | 0.55 | 0.52 | 0.76 | 0.74 |
| false causality | 167 | 0.59 | 0.43 | 0.42 | 0.47 | 0.65 | 0.65 |
| ad populum | 150 | 0.68 | 0.62 | 0.61 | 0.60 | 0.73 | 0.76 |
| circular reasoning | 131 | 0.56 | 0.58 | 0.52 | 0.49 | 0.51 | 0.55 |
| appeal to emotion | 127 | 0.43 | 0.37 | 0.38 | 0.24 | 0.49 | 0.50 |
| fallacy of logic | 119 | 0.33 | 0.32 | 0.31 | 0.21 | 0.35 | 0.39 |
| fallacy of relevance | 111 | 0.24 | 0.25 | 0.25 | 0.22 | 0.36 | 0.49 |
| intentional | 110 | 0.27 | 0.27 | 0.29 | 0.10 | 0.29 | 0.30 |
| false dilemma | 109 | 0.62 | 0.74 | 0.82 | 0.70 | 0.82 | **0.85** |
| fallacy of credibility | 107 | 0.30 | 0.18 | 0.19 | 0.14 | 0.45 | 0.47 |
| fallacy of extension | 100 | 0.58 | 0.57 | 0.57 | 0.48 | 0.59 | 0.60 |
| equivocation | 37 | 0.33 | 0.07 | 0.11 | 0.00 | 0.28 | 0.28 |

**What it shows:**
- **Easy classes have recognisable wording:** false dilemma ("either…or"), ad populum ("everyone"), ad hominem (insults).
- **Hard classes** are the rare *equivocation* and the vague, overlapping ones: *intentional*, and *fallacy of
  logic / relevance / credibility*.
- **Frequency is not everything.** False dilemma is mid-frequency but the easiest class.

Files: `results/per_class_f1_strict.csv`. Figure: `figures/final/per_class_f1.png`.

### 11.4 Class weighting (RQ3)

| Model | Plain | Class-weighted | Change |
|---|---|---|---|
| TF-IDF | 0.473 | 0.475 | +0.002 |
| CNN | 0.423 | 0.426 | +0.003 (std drops from 0.028 to 0.012) |
| BiLSTM | 0.358 | 0.333 | −0.025 (std rises to 0.091) |
| DistilBERT | 0.533 | 0.554 | +0.021 (not significant, p = 0.12) |

Class weighting never lifts *equivocation* or *intentional* above 0.30. Reweighting the loss cannot create the varied
examples the model lacks. Figure: `figures/final/class_weighting_effect.png`.

### 11.5 Official vs strict test

Every learned model scores higher on the official test than on the strict test:
- DistilBERT-cw: 0.570 → 0.554
- CNN: 0.441 → 0.423
- BiLSTM: 0.369 → 0.358

The 17 removed sentences had already been seen in train or dev. The subsets differ in composition, so the gap is not
a pure memorisation estimate, but it shows that reported LOGIC scores depend on how overlap is handled.

### 11.6 Climate transfer (RQ4)

Every model collapses without adaptation. The best macro-F1 is **0.146** (DistilBERT-cw); TF-IDF gets 0.127 and the
CNN 0.078. Climate text is journalism rather than quiz sentences, and its label distribution is very different, with
*intentional* the most common class. Jin et al. also reported a large drop (27.23% micro-F1 for their best model).

### 11.7 Dev curves

`figures/final/dev_curves.png`. DistilBERT reaches about 0.52 dev macro-F1 after just 2 epochs and peaks at about
0.63 by epochs 6–8. The CNN and BiLSTM climb more slowly and plateau lower (best epochs 10–20).

---

## 12. Error analysis and explanations

**Most common DistilBERT errors (seed 42, strict test):**
- faulty generalization → fallacy of logic (6)
- fallacy of relevance → fallacy of extension (6)
- faulty generalization → ad hominem (4)
- fallacy of logic → faulty generalization (4)
- appeal to emotion → faulty generalization (4)

Most confusions involve the broad "fallacy of …" categories, whose definitions overlap with more specific classes.
This points to taxonomy overlap more than to a weak model.

Confusion matrix: `figures/final/confusion_matrix_selected.png`.

**Occlusion probe** (`python -m fallacylens.probe`). Over correctly classified strict-test sentences, we counted each
sentence's two most influential words:
- **ad populum:** "everyone" appeared 11 times
- **false dilemma:** "or" appeared 7 times and "either" 3 times
- **circular reasoning:** "because" appeared 7 times
- **ad hominem:** insult words such as "idiot" and "hugger"

The models use the same surface cues a human would point to. This also shows **shortcut risk**: a sentence with
"everyone" might be called ad populum even when the reasoning is sound.

The occlusion method is implemented in `Predictor.explain()` in `fallacylens/predictor.py`. Only words inside the
model's input window are scored, so truncation can't distort the result.

---

## 13. How the project evolved

| Stage | What happened | Result |
|---|---|---|
| **Phase 1** (Sept 2026) | Literature review (10 papers), dataset audit, CNN baseline in `notebooks/phase1_fallacylens.ipynb` | CNN acc 0.528, macro-F1 0.459 on 299 deduplicated test rows. That protocol differs, so the number is historical. |
| **PR #1: pilot final experiments** | `train_final.py`. Introduced the leakage-controlled protocol, with a single seed and BERT-tiny (4M) as the "transformer". | Strict macro-F1: CNN 0.408, BERT-tiny tuned 0.407, BiLSTM 0.306, majority class 0.026. |
| **PR #2: visualizer** | Static animated HTML story page. Illustrative only, with no real model. | Replaced by the real web app. |
| **PR #3: final study (this work)** | Kept PR #1's protocol. Added TF-IDF and DistilBERT, class weighting for all, 3 seeds, significance tests, occlusion, the web app (ReUI), the IEEE paper and speaker notes. Code review with 10 findings, all addressed. | DistilBERT-cw 0.554 strict macro-F1. |
| **PR #4** | Word version of the paper | `paper/fallacylens_ieee.docx` |

**Why our CNN differs from Phase 1 (0.459 vs 0.423):**
- Different cleaning: conflicting labels are removed and dev overlap is removed from train.
- A different test set: strict, 283 rows.
- Mean over 3 seeds instead of one run. The CNN's seed spread is large (std 0.028).

**Why BERT-tiny in PR #1 scored lower than our DistilBERT:** BERT-tiny has 4M parameters and 2 layers, too small to
carry much pretrained knowledge. DistilBERT has 67M parameters and 6 layers.

---

## 14. The web application

- **Backend:** `app/server.py` (FastAPI).
- **Frontend:** `web/` (React 19 + Vite + Tailwind v4 + shadcn/ui + **ReUI** components: Badge, Alert, Stepper, Timeline).

**Tabs:**
1. **Analyze:**
   - Type a sentence or click **Load example** (a held-out strict-test sentence with its true label).
   - **Classify** shows the predicted fallacy, its confidence, the definition, the top-5 probabilities and **word
     influence** (blue supports, red works against).
   - **Compare all models** runs all 6 neural models and shows how many agree.
2. **Results:** summary cards, the full comparison table, per-class F1 bars and the significance table.
3. **Method:** a 5-step pipeline stepper, three architecture cards and a phase timeline.

Other features: a model picker in the header (default: DistilBERT cw) and a dark/light toggle. It works on phones.

**API endpoints:**

| Endpoint | What it returns |
|---|---|
| `GET /api/meta` | models, labels and definitions, default model |
| `POST /api/analyze {text, model}` | label, confidence, top-5, per-word weights |
| `POST /api/compare {text}` | prediction of every model |
| `GET /api/results` | `results/summary.json` |
| `GET /api/examples` | 2 strict-test examples per class |

**Live demo script:**
1. Load an example and classify it; point at the confidence and the highlighted words.
2. Type: *"Either we ban all cars or the city will choke on smog."* It should come back as false dilemma, around 99%.
3. Type: *"Everyone is buying it, so it must be good."* All 6 models agree on ad populum.
4. Open the Results tab and show the table and significance.
5. State the caveat: "The model always picks a fallacy. There is no 'sound argument' class."

---

## 15. Where everything is

| What | Path |
|---|---|
| **This guide** | `docs/MASTER_GUIDE.md` |
| Presentation script (4 parts) | `docs/SPEAKER_NOTES.md` |
| **IEEE paper (PDF / Word / LaTeX)** | `paper/fallacylens_ieee.pdf`, `paper/fallacylens_ieee.docx`, `paper/fallacylens_ieee.tex` |
| Paper references (BibTeX) | `paper/refs.bib` |
| Paper tables (auto-generated) | `paper/make_tables.py` → `paper/generated.tex`, `paper/tab_*.tex` |
| Word paper builder | `paper/docx/tex_to_blocks.py`, `paper/docx/build_docx.js` |
| Data protocol | `fallacylens/data.py` |
| Model definitions | `fallacylens/models.py` |
| Training (all experiments) | `fallacylens/train.py` |
| Analysis, figures, significance | `fallacylens/analyze.py` |
| Inference + occlusion | `fallacylens/predictor.py` |
| Occlusion probe | `fallacylens/probe.py` |
| Web API | `app/server.py` |
| Web UI | `web/src/App.tsx`, `web/src/components/*.tsx` |
| **Final results table** | `results/final_results.csv` |
| Per-class F1 | `results/per_class_f1_strict.csv` |
| Significance tests | `results/significance.csv` |
| Everything as JSON (used by the app) | `results/summary.json` |
| Data cleaning counts | `results/protocol.json` |
| Every run's metrics and predictions (24 files) | `results/runs/<exp>_s<seed>.json` |
| Figures | `figures/final/model_comparison.png`, `per_class_f1.png`, `confusion_matrix_selected.png`, `class_weighting_effect.png`, `dev_curves.png` |
| Trained models (local, gitignored, regenerable) | `models/<exp>/` |
| Phase 1 report + literature matrix | `report/PHASE1_REPORT.md`, `VERIFIED_REFERENCES.md` |
| Phase 1 notebook | `notebooks/phase1_fallacylens.ipynb` |
| PR #1 pilot (kept for reference) | `train_final.py`, `analyze_final.py`, `verify_final.py`, `demo.py`, `report/FINAL_REPORT.md`, `results/final/` |
| Raw data (downloaded automatically) | `data/raw/` |

GitHub: https://github.com/lakshitsachdeva/fallacydetection (PRs #1–#4 merged into `main`).

---

## 16. How to reproduce everything

```bash
git clone https://github.com/lakshitsachdeva/fallacydetection
cd fallacydetection
python -m venv .venv
.venv/Scripts/activate                 # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt        # for a GPU: install torch from pytorch.org first

python -m fallacylens.train            # 24 runs, ~20 min on a laptop GPU (CPU works, slower)
python -m fallacylens.analyze          # tables, figures, significance, results/summary.json
python -m fallacylens.probe            # occlusion word counts

cd web && npm install && npm run build && cd ..
python -m uvicorn app.server:app --port 8000   # open http://127.0.0.1:8000

python paper/make_tables.py && cd paper && tectonic fallacylens_ieee.tex   # rebuild paper PDF
```

The data is pinned by commit and hash, seeds are fixed, and every number in the paper is generated from
`results/summary.json`.

---

## 17. Limitations, ethics, future work

**Limitations:**
- LOGIC is small (1,801 training sentences) and quiz-style, so scores describe textbook examples.
- There is **no "not a fallacy" class**. Every input, including a sound argument, gets a fallacy label.
- Released rows are single-label, but real arguments can contain several fallacies.
- We used one pretrained size (DistilBERT) and 3 seeds. Larger models or more seeds could change the margins.
- Occlusion shows sensitivity, not reasoning.
- Models learn surface cues ("everyone", "or"), which is a shortcut risk.

**Ethics / responsible use:**
- This is a decision-support tool, **not a judge of truth**. The UI says so.
- It should not be used to automatically censor or label people.

**Future work:**
- Larger encoders such as RoBERTa or DeBERTa, and more seeds.
- Add a no-fallacy class, covering detection as well as classification.
- Domain adaptation and training on natural corpora (CoCoLoFa, MAFALDA).
- Structure-aware inputs, as in Jin et al.
- LLM prompting baselines.

---

## 18. Viva / examiner questions and answers

### Problem and motivation
**Q: What is a logical fallacy?**
An error in reasoning. The premises don't actually support the conclusion, even if they are true.

**Q: How is this different from fake-news detection or fact-checking?**
Fact-checking verifies whether a claim is true using outside evidence. We analyse whether the *reasoning* is valid,
using only the text. A true statement can be argued fallaciously.

**Q: Detection or classification?**
Classification, into 13 types. LOGIC has no "no fallacy" class.

**Q: Why is it hard?**
- The signal is reasoning structure, not keywords.
- The classes overlap (the "fallacy of …" group).
- The data is small and imbalanced.

### Data
**Q: Why LOGIC?**
- It is single-label and clean, with fixed splits.
- It is public at a pinned commit.
- It is the most-used multi-class benchmark, with published baselines.

The other datasets have incompatible taxonomies (MAFALDA has 23 classes and is multi-label, evaluation-only;
CoCoLoFa has 8 classes plus none).

**Q: What leakage did you find?**
- 5 training texts with conflicting labels.
- 25 duplicates within train.
- 13 training texts that also appear in dev.
- 17 test sentences already seen in train or dev.

**Q: What is the strict test and why use it?**
The 283 test sentences no model saw during training or selection. It measures generalisation, not memorisation.

**Q: How imbalanced is it?**
Faulty generalization has 314 examples and equivocation 37, roughly 8.5 : 1. That's why we headline macro-F1.

**Q: Did you remove stopwords?**
No. Words like *all*, *because* and *either/or* carry the reasoning signal, and occlusion confirmed the models use them.

### Models
**Q: Why a CNN?**
It is a classic fast sentence classifier: its filters detect short phrases. It is also our Phase 1 baseline.

**Q: Why a BiLSTM with attention?**
To test whether modelling word order and long-range context helps. Attention focuses on the key words.

**Q: Why DistilBERT and not BERT-large or an LLM?**
It is pretrained (which is the variable we test), runs on a 6 GB laptop GPU in about 2 minutes per run, and allows 3
seeds. Larger models are future work.

**Q: How many parameters does each model have?**
- TF-IDF: about 101K weights
- CNN: 437K
- BiLSTM: 584K
- DistilBERT: 67M

**Q: Explain the CNN layers.**
1. Embedding: word → 100-d vector.
2. Conv1d with widths 3, 4 and 5 (100 filters each), acting as n-gram detectors.
3. Global max-pool, which keeps each filter's strongest match.
4. Concatenation to 300 dimensions.
5. Dropout 0.5.
6. Linear layer to 13 outputs, then softmax.

**Q: Explain attention in your BiLSTM.**
Each hidden state gets a score from a small neural network: Linear → tanh → Linear. A softmax over time turns the
scores into weights, and the sentence vector is the weighted sum of the hidden states.

**Q: What is fine-tuning?**
We start from DistilBERT's pretrained weights, add a classification head, and train the whole network on LOGIC with a
small learning rate (5e-5).

**Q: What is class weighting?**
Loss weight w_c = N / (K · n_c). Rare classes get bigger weights, so mistakes on them cost more.

### Training and evaluation
**Q: How did you choose the final model?**
The highest mean dev macro-F1 across 3 seeds. The test set was not used.

**Q: How did you avoid overfitting?**
Dropout, early stopping on dev, a held-out strict test, weight decay and a short schedule for DistilBERT.

**Q: Why 3 seeds?**
Small-data results vary with the random seed; the CNN's std is 0.028. We report the mean ± std, not one lucky run.

**Q: Why macro-F1?**
It weights all 13 classes equally. Accuracy would hide failure on rare classes.

**Q: Your 0.55 is lower than Jin et al.'s 58.77%. Why?**
They report **micro-F1** (essentially accuracy) on the **official** test. We report **macro-F1** on a **strict** test.
Our comparable number is accuracy: 0.618 on the official test and 0.603 on the strict test, which is in the same
range as theirs.

**Q: What is a paired bootstrap?**
Resample the 283 test rows with replacement 2,000 times, recompute the score difference each time, and take the middle
95% as the confidence interval.

**Q: What is McNemar's test?**
It compares two classifiers on the same examples, using only the cases where they disagree.

**Q: Is DistilBERT significantly better?**
Yes against TF-IDF (p = 0.002), CNN and BiLSTM (p < 0.001). Weighted vs plain DistilBERT is not significant (p = 0.12).

### Results and insight
**Q: Main finding?**
Pretraining matters most on small data. TF-IDF beats both neural networks trained from scratch, and DistilBERT beats everything.

**Q: Why did TF-IDF beat the CNN and BiLSTM?**
1,801 sentences are too few to learn good word embeddings from scratch. TF-IDF has fewer things to learn and uses
n-gram cues directly.

**Q: Why is the BiLSTM worst?**
It has the most parameters to fit from scratch, its recurrent training is harder, and its seed variance is high.
Word-order modelling doesn't pay off at this size.

**Q: Did class weighting help?**
- DistilBERT: a small, non-significant gain (+0.021).
- CNN: essentially no change.
- BiLSTM: it hurt.

It doesn't fix rare classes because it cannot create missing examples.

**Q: Which classes are hardest and why?**
Equivocation (0.28; rarest, and it depends on word meaning) and intentional (0.30). The vague "fallacy of logic /
relevance / credibility" classes overlap with others.

**Q: Which are easiest?**
False dilemma (0.85, "either…or"), ad populum (0.76, "everyone") and ad hominem (0.74, insults).

**Q: What happened on climate news?**
Everything collapses, to a best of 0.146 macro-F1. The domain shift and the different label distribution break it.
Quiz-trained models don't generalise to real text.

**Q: What do the highlighted words mean?**
They come from occlusion: removing a blue word lowers the predicted probability. They show sensitivity, not proof of reasoning.

### Engineering
**Q: How does the app work?**
React frontend → FastAPI backend → PyTorch models loaded once and cached → JSON response. Explanations are
computed per request by batched occlusion.

**Q: What is ReUI?**
A component library built on shadcn/ui and Base UI. We used its Badge, Alert, Stepper and Timeline.

**Q: Is it reproducible?**
Yes. The data is pinned with hash checks, seeds are fixed, the commands are in Section 16, and every per-run JSON is saved.

**Q: Did you review the code?**
Yes. A structured code review found 10 issues, including stale UI results, the occlusion truncation window, cache
refresh and safe checkpoint loading (`weights_only=True`). All were fixed or verified.

### Critical questions
**Q: Isn't this just text classification?**
Architecturally, yes. The research content is the task, the leakage audit, the controlled comparison, the per-class
and transfer analysis, and the significance testing.

**Q: Could the model be using shortcuts?**
Yes. Occlusion shows it relies on cue words like "everyone" and "or". That is why we report it as a limitation and
tested transfer, which exposes shortcuts.

**Q: What would you do with more time?**
- Larger encoders and more seeds.
- A no-fallacy class.
- Domain adaptation on CoCoLoFa or MAFALDA.
- Structure-aware inputs.

---

## 19. Numbers cheat sheet

- Data: 1,849 / 300 / 300 original rows → **1,801 train / 300 dev / 300 official test / 283 strict test / 211 climate**
- Cleaning: 5 conflicting-label texts, 17 test rows seen before
- 13 classes; largest 314, smallest 37 (≈ 8.5 : 1)
- Median sentence length 20 words; max length 54 tokens (CNN/BiLSTM), 128 (DistilBERT)
- 8 configurations × 3 seeds = **24 runs**, about 20 min total on an RTX 3050
- **Strict macro-F1:**
  - **DistilBERT-cw 0.554**
  - DistilBERT 0.533
  - TF-IDF 0.473
  - CNN 0.423
  - BiLSTM 0.358
- Strict accuracy, DistilBERT-cw: **0.603**; official-test accuracy: 0.618
- Significance: vs TF-IDF p = 0.002; vs CNN and BiLSTM p < 0.001; vs plain DistilBERT p = 0.12
- Best class: false dilemma 0.85. Worst: equivocation 0.28.
- Climate transfer: best macro-F1 0.146
- Literature: Jin et al. structure-aware 58.77% micro-F1 (official test); zero-shot 12.22%; climate transfer 27.23%
- Phase 1 CNN: 0.459 macro-F1 (old protocol). PR #1 pilot CNN: 0.408 (single seed).
