# FallacyLens Model Story

An offline-first, presentation-ready visualizer for the professor walkthrough.
It explains the project as a story rather than a metrics dashboard:

1. raw argument → token representation → structure signal → human-readable hypothesis
2. CNN vs BiLSTM vs Transformer as three different model lenses
3. an animated, local input lab with transparent illustrative cues
4. a responsible-use frame: model signal is not a truth verdict

The page intentionally does **not** display aggregate accuracy, F1, or final
model scores. Those belong in the technical report; this page is for explaining
how the models work and how a prediction should be interpreted.

## Preview

From the repository root:

```bash
python -m http.server 8000 -d visualizer
```

Then open `http://localhost:8000`.

## Presentation controls

- **Present**: toggles a projection-friendly mode and attempts fullscreen.
- **Motion on/off**: lets the presenter reduce animation during discussion.
- **1 / 2 / 3**: jump to the CNN, BiLSTM, or Transformer architecture card.
- **Animate the analysis**: runs the local explanatory trace in the lab.
- **Replay from the beginning**: returns to the hero story.

## Integrity note

The browser lab uses hand-authored illustrative cues so the visualizer remains
portable and does not pretend to be a hidden replacement for the trained Python
models. Real inference remains available through the project’s Python demo and
experiment artifacts. This separation keeps the presentation compelling while
keeping research claims honest.
