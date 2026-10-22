# FallacyLens project completion record

**Date:** 22 October 2026
**Status:** Phases 1–3 implementation, measured evaluation, report, notebook, and demo complete.

- Phase 1 literature review and CNN baseline remain in `PHASE1_REPORT.md` and `notebooks/phase1_fallacylens.ipynb`.
- Final comparison trained six experiments using the pinned official LOGIC data and BERT-tiny revision. The dev-selected model is the CNN. Its strict LOGIC test macro F1 is **0.4075** on 283 text-exclusive rows; the tuned Transformer has **0.4068**. Direct LOGICCLIMATE transfer remains weak (best macro F1 **0.0976**).
- `verify_final.py` independently recomputed all 24 model/split metric rows from saved predictions and confirmed dev-only model selection.
- `notebooks/final_fallacylens.ipynb` executed all seven code cells with no errors. `demo.py` ran and returned top-three labels from the saved CNN checkpoint.
- Final report, presentation guide, aggregate CSV/JSON results, and final figures exist. `results/final/protocol.json` records source hashes, dataset counts, label order, seed, and model revision.

**Scope of the claim:** This is a completed student research project package, not a validated general-purpose fallacy judge. It uses one training seed, the released LOGIC label taxonomy, and a small held-out test. The climate transfer result is poor. No academic submission or faculty review is claimed.
