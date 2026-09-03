# Push Instructions — FallacyLens Phase 1

Everything for Phase 1 is committed locally on branch `main` (see `git log`). The remote is
configured as `origin = https://github.com/lakshitsachdeva/fallacydetection.git`.

**Why this file exists**: the automated run had no GitHub credentials (`gh auth status` → not
logged in; no `GITHUB_TOKEN` env var), and the task spec forbids creating credentials. So nothing
has been pushed.

## What the owner needs to do (one-time, ~2 minutes)

```bash
cd ~/fallacydetection

# Option A — GitHub CLI:
gh auth login          # follow the prompts (choose HTTPS + your account)
git push -u origin main

# Option B — personal access token:
# create a token at https://github.com/settings/tokens (repo scope),
# then either:
git push -u origin main
#   (enter your GitHub username and the token as password when prompted)
# or:
#   git remote set-url origin https://<TOKEN>@github.com/lakshitsachdeva/fallacydetection.git
#   git push -u origin main
```

## After pushing

- Verify on https://github.com/lakshitsachdeva/fallacydetection that `report/`,
  `notebooks/`, `results/`, `figures/` are visible.
- `.venv/`, `logs/`, `data/raw/`, and `*.pt` checkpoints are gitignored and will not be pushed
  (the notebook re-downloads the dataset automatically when run).
