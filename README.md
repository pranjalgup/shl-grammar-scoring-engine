# SHL Grammar Scoring Engine

Predicts a continuous grammar score (0-5) from 45-60s spoken-audio clips,
for the SHL Hiring Assessment 2026 Kaggle competition.

## Approach

1. **ASR transcription** (`scripts/transcribe.py`) — Whisper (`small.en`, via
   `faster-whisper`) transcribes each clip with word-level timestamps.
2. **Feature extraction** (`scripts/extract_features.py`) — prosodic features
   from Whisper timing/confidence, syntactic features from spaCy (POS ratios,
   fragment/run-on detection, parse depth), and a GPT-2 perplexity-based
   fluency score.
3. **Modeling** (`scripts/train_model.py`, and the full writeup in
   `Grammar_Scoring_Engine.ipynb`) — Ridge / SVR / Gradient Boosting compared
   via 5-fold cross-validation; Ridge selected as the best, most
   interpretable model.

## Results

- Training RMSE: 0.744
- 5-fold CV RMSE: 0.785, Pearson r: 0.774

See `Grammar_Scoring_Engine.ipynb` for the full EDA, diagnostics, and
discussion of approach/limitations.

Note: the raw audio dataset (`data/Dataset_Final/`) is excluded from this
repo (per Kaggle competition rules on private data) — only the derived,
non-identifying transcripts/features/predictions are included so the
pipeline's outputs are reproducible and inspectable.
