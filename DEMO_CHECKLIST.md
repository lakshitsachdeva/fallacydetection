# Pre-Demo Checklist — FallacyLens Phase 1

Run through this the morning of the review.

## Artifacts exist
- [ ] `report/PHASE1_REPORT.md` — complete (parts 1–4, 12, 13, matrix, references)
- [ ] `notebooks/phase1_fallacylens.ipynb` — executed, outputs visible
- [ ] `results/phase1_results.csv` — CNN baseline row with real numbers
- [ ] `figures/` — class distribution, text-length plot, training curves, confusion matrix
- [ ] Repo pushed to GitHub: https://github.com/lakshitsachdeva/fallacydetection

## Numbers check (5 min)
- [ ] Every `{{FILL: ...}}` slot in `PRESENTATION_GUIDE.md` replaced with real values
- [ ] Results table numbers match the executed notebook exactly
- [ ] Literature matrix numbers match the cited papers (spot-check 2 rows)
- [ ] Nothing labeled "preliminary" is described as final anywhere

## Repro check (optional but impressive)
- [ ] Fresh shell: `pip install -r requirements.txt` then
      `jupyter nbconvert --to notebook --execute notebooks/phase1_fallacylens.ipynb`
      completes without errors
- [ ] Seed cell shows seed = 42

## On the day
- [ ] Laptop charged, notebook + report + figures open in tabs before you enter
- [ ] You can say the 60-second pitch without notes
- [ ] You know your macro F1 and what it means
- [ ] You can name 2 limitations and 2 Phase 2 items without looking
