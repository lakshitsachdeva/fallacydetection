# FallacyLens final presentation quick guide

## 60-second walkthrough

“FallacyLens classifies short arguments into the 13 labels in the LOGIC research dataset. Phase 1 reviewed prior work and built a CNN baseline. For the final project, I audited duplicate text and conflicting labels, then trained a fresh CNN, BiLSTM, and compact pretrained Transformer on 1,801 cleaned training examples. I also tested class weighting, a frozen encoder, and a faster Transformer fine-tuning schedule. The CNN was selected using development data. On a strict 283-example test with previously seen text excluded, it reached 0.4075 macro F1; the tuned Transformer reached 0.4068, effectively tied within sampling uncertainty. Direct transfer to climate claims was poor for every model, at 0.0525 to 0.0976 macro F1. The lesson is that an in-domain fallacy label is much easier than reliable general fallacy reasoning.”

## Live demo order

1. Open [the final report](report/FINAL_REPORT.md) at the comparison table, then show [the executed final notebook](notebooks/final_fallacylens.ipynb).
2. Show the strict-test confusion matrix in `figures/final/confusion_matrix.png`: equivocation recall is 0/5 and credibility recall is 1/17.
3. Run `.venv/bin/python demo.py "Most people agree with me, so I must be right."` from the repo root. Explain that the displayed scores are uncalibrated model outputs among 13 labels, not judgments of truth.
4. Show the climate comparison chart in `figures/final/comparison.png` and the paired interval in `results/final/uncertainty.json`.

## Questions to be ready for

- **Why macro F1?** It averages F1 across classes, so the smallest classes count rather than being drowned out by frequent labels.
- **Why is Phase 1's 0.4594 CNN score different?** Phase 1 used a different preprocessing and overlap protocol; its score is historical context. The final CNN was retrained under the same protocol as all final models.
- **Did the Transformer win?** The dev-selected CNN and tuned Transformer are almost tied on strict LOGIC test: 0.4075 vs 0.4068 macro F1. The paired row-bootstrap interval for their difference crosses zero widely.
- **Did class weighting help?** It improved the original Transformer schedule's strict macro F1 from 0.1635 to 0.1946, but remained below the CNN and tuned Transformer.
- **Can the model identify real misinformation?** No. It classifies among the dataset's fallacy labels and has no no-fallacy, factual verification, or argument-context component.
- **What would you improve next?** Repeat seeds, improve minority-class examples, test stronger encoders under a fixed budget, and develop with a separate target-domain dev set before claiming climate transfer.
