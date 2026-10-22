# FallacyLens — Master Speaker Notes (4 presenters)

**Talk length:** about 16 minutes plus Q&A. Each presenter speaks for about 4 minutes.
**Keep open before you start:** the web app (http://127.0.0.1:8000), the paper PDF
(`paper/fallacylens_ieee.pdf`) and `figures/final/`.

| Part | Presenter | Topic | Time | Show on screen |
| --- | --- | --- | --- | --- |
| 1 | Presenter A | Problem, motivation, prior work, research gap | 0:00–4:00 | Title slide, one example argument, related-work summary |
| 2 | Presenter B | Dataset, leakage audit, evaluation protocol | 4:00–8:00 | Pipeline figure (paper Fig. 1), class counts |
| 3 | Presenter C | Models, training setup, live demo | 8:00–12:00 | Architecture figure (Fig. 2), web app Analyze tab |
| 4 | Presenter D | Results, significance, per-class, transfer, limits, conclusion | 12:00–16:00 | Results tab, Table I, per-class heatmap, conclusion slide |

All numbers below are from `results/final_results.csv` and `results/significance.csv`
(mean over seeds 42, 43, 44, on the strict test unless stated otherwise).

---

## Part 1 — Presenter A: The problem and why it matters (4 min)

**Opening (30 s)**
> "Here is an argument: *Everyone in my class uses this phone, so it must be the best one.*
> Nothing in it is false, so a fact-checker has nothing to correct. The problem is the step
> from 'popular' to 'best'. That step is called a logical fallacy, in this case *ad populum*.
> Our project, FallacyLens, trains models to name the fallacy in a short argument."

**What a fallacy is (45 s)**
- A fallacy is a flaw in reasoning. The argument can sound convincing even when every premise is true.
- Contrast it with fact-checking. Fact-checking asks "is this claim true?". We ask "does this reasoning hold up?".
- Our task is **classification**: we pick one of 13 fallacy types for each argument.
  Example types: ad hominem, false dilemma, false causality, circular reasoning.

**Why it matters (45 s)**
- Fallacies are common in online debate, advertising and political speech.
- Possible uses are teaching critical thinking, giving feedback on debates and helping moderators.
- Say clearly: "This is a decision-support tool. It is not a judge of truth."

**Prior work (1 min)**
- Jin et al. (EMNLP Findings 2022) released **LOGIC**, about 2.4k quiz-style sentences with 13 labels,
  and **LOGICCLIMATE**, sentences taken from climate news. That is our dataset.
- Later work moved to new corpora (MAFALDA, CoCoLoFa) and to large language models with prompting.
  MAFALDA found that humans still beat every LLM they tested.
- Point to note: papers use different metrics and treat duplicated text differently, so their
  numbers are hard to compare.

**Research gap and our questions (1 min)**
> "Nobody had compared the standard model families on LOGIC under one strict, leakage-free protocol.
> So we asked three questions."
1. How do TF-IDF, CNN, BiLSTM and a pretrained transformer (DistilBERT) compare?
2. Does class weighting help the rare fallacy types?
3. Do the models still work on a different domain, climate news?

**Handover:** "To answer these fairly, we first had to clean the data. Presenter B explains how."

---

## Part 2 — Presenter B: Data and a fair evaluation (4 min)

**The dataset (1 min)**
- LOGIC has 1,849 / 300 / 300 train / dev / test sentences and 13 classes.
- Sentences are short, about 20 words at the median.
- The classes are imbalanced. *Faulty generalization* has 314 training examples and *equivocation* only 37.
  That is roughly 8 to 1.
- This is why our headline metric is **macro-F1**, which weights every class equally.
  Plain accuracy would hide failure on the rare classes.

**What we found when we audited the data (1.5 min)**
> "Before training, we compared every sentence against every other one."
- **5 training texts appear twice with different labels.** The text cannot tell us which label is right, so we removed them.
- Repeated training sentences were collapsed into one copy.
- Some training sentences also appeared in the dev set. We removed them, so early stopping cannot be fooled by memorised sentences.
- **17 test sentences had already appeared in train or dev.** We keep the official test for reference, but our
  main score uses a **strict test of 283 sentences** that no model has seen.
- Result: 1,801 clean training sentences.
- Show the pipeline figure (paper Fig. 1).

**Why this matters (45 s)**
- A model can score well on a sentence it has memorised. Every model scores higher on the official test
  than on the strict one, so reported LOGIC numbers depend on how overlap is handled.
- The data is downloaded from a pinned commit and checked with SHA-256 hashes. Anyone can reproduce it exactly.

**Preprocessing (45 s)**
- We keep punctuation and stopwords. Words like *all*, *therefore* and *either…or* are often the clue to the fallacy.
- The vocabulary is built from training text only.
- Test labels are never used for training, early stopping or model selection.

**Handover:** "With clean data in place, Presenter C will walk through the models and show the live system."

---

## Part 3 — Presenter C: The models and a live demo (4 min)

**Four model families (2 min)** — show paper Fig. 2
- **TF-IDF + logistic regression**: counts word and word-pair features. A simple, non-neural baseline.
- **CNN** (Kim, 2014): filters of width 3, 4 and 5 slide over word embeddings and pick up short cue phrases.
  It has 437K parameters and the embeddings are learned from scratch.
- **BiLSTM with attention**: reads the sentence in both directions, and attention picks the important words. It has 584K parameters.
- **DistilBERT**: a 66M-parameter transformer already pretrained on general English, which we fine-tune on LOGIC.
- Every model is also trained **with class weighting**. Errors on rare classes cost more in the loss.

**Training setup (45 s)**
- Same data, same splits and same selection rule for every model: keep the epoch with the best dev macro-F1.
- Three random seeds (42, 43, 44) per configuration, so we report mean ± standard deviation, not one lucky run.
- DistilBERT takes about 2 minutes per run on a laptop RTX 3050. The CNN takes about 10 seconds.

**Live demo (1.25 min)** — web app, Analyze tab
1. Click **Load example**. The app shows a held-out test sentence and its true label.
2. Click **Classify** with "DistilBERT (cw)" selected (the default). Point at the predicted fallacy, its confidence and the top-5 bars.
3. Point at **Word influence**. Blue words support the prediction and red words work against it.
   Across the test set, "everyone" drives ad populum, "or" and "either" drive false dilemma, and "because" drives circular reasoning.
   "We delete each word in turn and measure how much the probability drops."
4. Click **Compare all models**. "Here you see how often the models agree."
5. If time allows, type your own sentence, for example:
   *"Either we ban all cars or the city will choke on smog."* (a false dilemma).
- Say the caveat: "The model always picks one of 13 fallacies. There is no 'sound argument' option."

**Handover:** "So how well do these models actually do? Presenter D has the numbers."

---

## Part 4 — Presenter D: Results and conclusions (4 min)

**Main result (1.25 min)** — Results tab or paper Table I
- Strict-test macro-F1, mean of three seeds:
  - **DistilBERT with class weighting: 0.554**, the best model, also chosen by the dev set
  - DistilBERT without weighting: 0.533
  - TF-IDF + LR: 0.473
  - CNN: 0.423
  - BiLSTM-attention: 0.358
- Key message: "**Pretraining is what makes the difference.** A simple TF-IDF model beats both neural networks
  trained from scratch. With only 1,800 sentences, the networks cannot learn good word representations on their own."

**Is it real? (45 s)** — significance table
- We used a paired bootstrap and McNemar's test on the same test sentences.
- DistilBERT vs the CNN and BiLSTM: the confidence intervals exclude zero and p < 0.01, so the difference is real.
- DistilBERT vs TF-IDF: ahead by 0.09 macro-F1, p = 0.002, so this difference is significant too.
- Weighted vs plain DistilBERT: p = 0.12, not significant. "Class weighting is a small refinement, not a separate win."

**Where models fail (1 min)** — per-class heatmap
- Easy classes have clear wording. *False dilemma* (either/or) scores 0.85, and *ad populum* and *ad hominem* score about 0.75.
- Hard classes:
  - *Equivocation* (0.28) is the rarest class, and *intentional* (0.30) is also weak.
  - The vague "fallacy of logic / relevance / credibility" and *intentional* labels overlap with other classes.
- **Class weighting did not fix this.** It gave DistilBERT a small gain, barely changed the CNN and hurt the BiLSTM.
  Reweighting the loss cannot invent examples that are not in the data.
- **Climate news:** every model collapses, to macro-F1 of 0.146 at best. Training on quiz sentences does not
  carry over to real journalism.

**Limitations (30 s)**
- Small, quiz-style dataset, and no "not a fallacy" class.
- One pretrained model size and three seeds.
- Word influence shows sensitivity, not reasoning.

**Conclusion (30 s)**
> "We built a clean, reproducible benchmark for fallacy classification and showed three things.
> Pretrained transformers clearly beat CNN and BiLSTM models on small data. Class weighting is not enough
> for rare fallacies. And no model transfers to real-world climate text. Next steps are larger models, a
> non-fallacy class, and domain adaptation. Thank you, we are happy to take questions."

---

## Q&A — who answers what

| Likely question | Answer owner | Short answer |
| --- | --- | --- |
| Why macro-F1 and not accuracy? | B | Classes are imbalanced 8:1. Accuracy hides failure on rare classes. We report both. |
| Why is your number lower than Jin et al.'s 58.77%? | B | They report micro-F1 on the official test. We report macro-F1 on a strict test with seen sentences removed. The metric and the test set both differ. |
| Why did TF-IDF beat the CNN? | C | 1,800 sentences are too few to learn embeddings from scratch. TF-IDF has far fewer parameters to fit. |
| Why DistilBERT and not BERT-large or GPT? | C | It runs on a laptop GPU with three seeds in minutes. Larger models are future work. |
| Is the DistilBERT win statistically significant? | D | Yes, against TF-IDF (p = 0.002), the CNN and the BiLSTM (p < 0.001). |
| What does class weighting do? | C | It multiplies the loss for rare classes by N / (K · n_c). Here it gave small, inconsistent changes. |
| Can this detect fake news? | A | No. Fact-checking verifies claims. We analyse the reasoning pattern. |
| Can I trust the word highlights? | D | They show which words change the output, not why the argument is wrong. Use them for debugging. |
| What happens with a sound argument? | A / D | It still gets one of 13 fallacy labels. There is no "none" class, which is a key limitation. |
| How do I reproduce it? | B | `python -m fallacylens.train` then `python -m fallacylens.analyze`. The data is pinned by commit and hash. |

## Rehearsal checklist
- [ ] Everyone can state the headline numbers without notes: DistilBERT (cw) 0.554, TF-IDF 0.473, CNN 0.423, BiLSTM 0.358.
- [ ] Presenter C has the app running and one example sentence ready to paste.
- [ ] Each handover sentence is practised, so the talk sounds like one story.
- [ ] Each part takes 4 minutes or less; time it once with a phone timer.
