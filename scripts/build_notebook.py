"""Assemble the final deliverable notebook (Grammar_Scoring_Engine.ipynb).

Cells are plain strings here for clarity; this script just wires them into
an .ipynb file. The notebook is designed to be re-run top to bottom:
expensive steps (ASR transcription) are cached to CSV and will only
recompute if the cache is missing, which is called out explicitly in the
markdown so a reviewer isn't surprised by a ~2hr first run.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# ---------------------------------------------------------------------------
md(r"""
# Grammar Scoring Engine — SHL Hiring Assessment 2026

**Author:** Pranjal Gupta

## 1. Problem

Given a spoken-audio clip (45-60s), predict a continuous **Grammar score
(0-5)** matching a human MOS-Likert rubric that grades sentence structure,
syntax, grammatical control, and handling of complex language.

Training set: 769 labeled clips. Test set: 216 clips to predict.
Evaluation: **Pearson correlation** and **RMSE** against hidden test labels.

## 2. Approach, in one paragraph

Grammar is a property of *language*, not of a raw waveform, so the pipeline
is **ASR first**: transcribe each clip to text with Whisper, then compute
interpretable linguistic + prosodic features from the transcript and from
Whisper's word-level timing/confidence output, and finally fit a small,
explainable regression model (Ridge / Gradient Boosting / SVR, compared via
cross-validation) on those features. With only 769 labeled examples, a
feature-based model that we can inspect (coefficients / importances) was
preferred over an end-to-end deep audio model, which would need far more
data to avoid learning spurious shortcuts (e.g. speaker identity, recording
quality) instead of grammar itself.

## 3. Pipeline architecture

```
audio (.wav) --[Whisper small.en, word timestamps]--> transcript + timing
                                                           |
                    +--------------------------------------+-------------------------------+
                    |                                       |                               |
            spaCy (en_core_web_sm)                 GPT-2 perplexity               Whisper-derived prosody
       POS ratios, fragment/run-on rate,         (fluency / predictability        speech rate, pause ratio,
       no-verb-sentence rate, parse depth,              of the word sequence)     ASR confidence, disfluency
       type-token ratio                                                           markers, repeated words
                    |                                       |                               |
                    +--------------------------------------+-------------------------------+
                                                           |
                                               34 engineered features
                                                           |
                                     Ridge / GBM / SVR regression (5-fold CV)
                                                           |
                                          Grammar score prediction (0-5, clipped)
```

## 4. Notebook structure

1. Data loading & EDA (label distribution, audio duration, example transcripts)
2. ASR transcription (cached — see note below)
3. Feature engineering (cached)
4. Feature-level EDA (correlation with label)
5. Model comparison via 5-fold cross-validation
6. Final model fit, **RMSE on training data (required)**, Pearson correlation, diagnostics
7. Feature importance / interpretability
8. Test-set predictions → submission file
9. Limitations & what I'd do with more time
""")

# ---------------------------------------------------------------------------
md("## 5. Setup")
code(r"""
import os
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

DATA = "data"
AUDIO_DIR = os.path.join(DATA, "Dataset_Final")
TRANSCRIPTS_CSV = os.path.join(DATA, "transcripts.csv")
FEATURES_CSV = os.path.join(DATA, "features.csv")

plt.rcParams["figure.dpi"] = 110
RANDOM_STATE = 42
""")

# ---------------------------------------------------------------------------
md(r"""
## 6. Data loading & exploratory data analysis

769 training clips with labels, 216 test clips. Labels are a 0-5 MOS-Likert
grammar score (participants could be scored at half-point granularity, e.g.
2.5, 3.5).
""")
code(r"""
train_labels = pd.read_csv(os.path.join(AUDIO_DIR, "train.csv"))
test_labels = pd.read_csv(os.path.join(AUDIO_DIR, "test.csv"))

print("Train:", train_labels.shape, " Test:", test_labels.shape)
train_labels.head()
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(11, 4))

train_labels["label"].value_counts().sort_index().plot(kind="bar", ax=ax[0], color="#4C72B0")
ax[0].set_title("Grammar score distribution (train)")
ax[0].set_xlabel("Score")
ax[0].set_ylabel("Count")

ax[1].hist(train_labels["label"], bins=20, color="#55A868")
ax[1].axvline(train_labels["label"].mean(), color="red", linestyle="--",
              label=f"mean={train_labels['label'].mean():.2f}")
ax[1].set_title("Grammar score histogram (train)")
ax[1].legend()
plt.tight_layout()
plt.savefig("plots/01_label_distribution.png")
plt.show()

print(train_labels["label"].describe())
""")

md(r"""
Observation: scores are concentrated around 3-5 with a smaller cluster at
0 (likely silent/corrupted/very poor clips) and almost nothing at 1-1.5.
This is a moderately imbalanced regression target — worth keeping in mind
when interpreting RMSE (errors on the rare low-score clips contribute
disproportionately to perceived model quality).
""")

# ---------------------------------------------------------------------------
md(r"""
## 7. ASR transcription (Whisper)

We use `faster-whisper`'s **small.en** model (int8, CPU) with word-level
timestamps and VAD filtering. This step is **cached to `data/transcripts.csv`**
— transcribing all 985 clips from scratch takes roughly 2 hours on a CPU-only
laptop, so the cell below only recomputes if that cache is missing (e.g. on
a fresh clone of the repo). The transcription code itself (`scripts/transcribe.py`)
is included in the submission for inspection.
""")
code(r"""
if os.path.exists(TRANSCRIPTS_CSV):
    transcripts = pd.read_csv(TRANSCRIPTS_CSV)
    print(f"Loaded cached transcripts: {transcripts.shape}")
else:
    print("No cache found — running full transcription (~2hr on CPU)...")
    import subprocess
    subprocess.run(["python", "scripts/transcribe.py"], check=True)
    transcripts = pd.read_csv(TRANSCRIPTS_CSV)

transcripts.head()
""")

code(r"""
# Example transcripts at different label levels, to build intuition for what
# the rubric is actually grading.
merged_preview = transcripts.merge(train_labels, on="filename", how="inner")
for score in [0.0, 2.0, 3.0, 5.0]:
    sub = merged_preview[merged_preview.label == score]
    if len(sub):
        ex = sub.iloc[0]
        print(f"--- label={score} ({ex.filename}) ---")
        print(ex.text[:280])
        print()
""")

md(r"""
Reading these side by side gives a feel for the rubric: a score-5 transcript
reads as fluent, complete sentences with subordinate clauses; low scores show
fragments, missing verbs, or (for the 0.0 cluster) near-empty / garbled
transcriptions — likely clips with very little intelligible speech.
""")

# ---------------------------------------------------------------------------
md(r"""
## 8. Feature engineering

Three feature families, computed in `scripts/extract_features.py` (cached to
`data/features.csv`):

**Prosodic (from Whisper timing/confidence)**
- `speech_rate_wps`, `articulation_rate_wps` — words/sec including vs. excluding pauses
- `pause_ratio`, `max_pause_sec` — hesitation signal (rubric mentions self-correction / incomplete sentences)
- `avg_word_conf` — Whisper's own ASR confidence (low confidence often means disfluent or unclear speech)

**Syntactic / grammar (spaCy `en_core_web_sm`)**
- POS-tag ratios (`pos_NOUN`, `pos_VERB`, `pos_AUX`, ... )
- `frag_sentence_ratio`, `runon_sentence_ratio`, `no_verb_sentence_ratio` — direct proxies for the rubric's "incomplete sentences" / "limited control of structure"
- `max_parse_depth` — syntactic complexity (rubric rewards "complex language structures")
- `type_token_ratio` — vocabulary variety

**Fluency**
- `gpt2_perplexity` / `gpt2_log_perplexity` — how "surprising" the transcript is to a pretrained language model; disfluent or ungrammatical text tends to be less predictable
- `disfluency_rate`, `repeated_bigram_rate` — filler words and word repetition

This step is cached the same way as transcription (recomputes if the cache
file is missing; a full rerun on 985 short transcripts takes a few minutes).
""")
code(r"""
if os.path.exists(FEATURES_CSV):
    features = pd.read_csv(FEATURES_CSV)
    print(f"Loaded cached features: {features.shape}")
else:
    print("No cache found — extracting features...")
    import subprocess
    subprocess.run(["python", "scripts/extract_features.py"], check=True)
    features = pd.read_csv(FEATURES_CSV)

feature_cols = [c for c in features.columns if c not in ("split", "filename")]
print(f"{len(feature_cols)} features:", feature_cols)
features.head()
""")

# ---------------------------------------------------------------------------
md("## 9. Feature-level EDA: correlation with grammar score")
code(r"""
train_feats = features[features.split == "train"].merge(train_labels, on="filename", how="inner")
test_feats = features[features.split == "test"].merge(test_labels[["filename"]], on="filename", how="inner")

corrs = train_feats[feature_cols + ["label"]].corr()["label"].drop("label").sort_values()

fig, ax = plt.subplots(figsize=(7, 9))
colors = ["#C44E52" if v < 0 else "#4C72B0" for v in corrs.values]
ax.barh(corrs.index, corrs.values, color=colors)
ax.set_title("Pearson correlation of each feature with grammar label (train)")
ax.set_xlabel("correlation")
plt.tight_layout()
plt.savefig("plots/02_feature_correlations.png")
plt.show()
""")

code(r"""
# A closer look at the strongest individual signals
top_feats = corrs.abs().sort_values(ascending=False).head(4).index.tolist()

fig, axes = plt.subplots(1, len(top_feats), figsize=(4 * len(top_feats), 4))
for ax, feat in zip(axes, top_feats):
    ax.scatter(train_feats[feat], train_feats["label"], alpha=0.4, s=15, color="#4C72B0")
    r, _ = pearsonr(train_feats[feat], train_feats["label"])
    ax.set_title(f"{feat}\n(r={r:.2f})")
    ax.set_xlabel(feat)
    ax.set_ylabel("label")
plt.tight_layout()
plt.savefig("plots/03_top_feature_scatter.png")
plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 10. Modeling: 5-fold cross-validated comparison

With only 769 labeled examples we deliberately stick to simple, regularized
models and validate with **5-fold CV** rather than a single holdout split
(a single split on this few examples would give a noisy, unreliable estimate
of generalization). Three candidates:

- **Ridge regression** — linear, heavily regularized, fully interpretable via coefficients
- **SVR (RBF kernel)** — allows non-linear feature interactions
- **Gradient Boosting** — non-linear, gives feature importances, more prone to overfitting at this sample size

All features are standardized before Ridge/SVR (tree-based GBM doesn't need it).
""")
code(r"""
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

def build_models():
    return {
        "ridge": Pipeline([("scaler", StandardScaler()),
                            ("reg", Ridge(alpha=5.0, random_state=RANDOM_STATE))]),
        "svr_rbf": Pipeline([("scaler", StandardScaler()),
                              ("reg", SVR(kernel="rbf", C=3.0, epsilon=0.2))]),
        "gbm": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                          subsample=0.8, random_state=RANDOM_STATE),
    }

X = train_feats[feature_cols].values
y = train_feats["label"].values
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_results = []
oof_preds = {}
for name in build_models():
    oof = np.zeros(len(y))
    for tr_idx, val_idx in kf.split(X):
        m = build_models()[name]
        m.fit(X[tr_idx], y[tr_idx])
        oof[val_idx] = m.predict(X[val_idx])
    r = rmse(y, oof)
    corr, _ = pearsonr(y, oof)
    cv_results.append({"model": name, "cv_rmse": r, "cv_pearson_r": corr})
    oof_preds[name] = oof

cv_df = pd.DataFrame(cv_results).sort_values("cv_rmse")
cv_df
""")

# ---------------------------------------------------------------------------
md(r"""
## 11. Final model, required RMSE, and diagnostics

We select the model with the best (lowest) cross-validated RMSE above, refit
it on the **full training set**, and report:

- **In-sample training RMSE** (required by the assignment)
- **Out-of-fold CV RMSE** — the honest, leakage-free estimate of generalization
  (in-sample RMSE is optimistic since the model has seen those labels during
  fitting; CV RMSE is what should be compared to the hidden test RMSE)
- **Pearson correlation** (the other leaderboard metric)
""")
code(r"""
best_name = cv_df.iloc[0]["model"]
print("Best model (by CV RMSE):", best_name)

final_model = build_models()[best_name]
final_model.fit(X, y)
train_preds_insample = final_model.predict(X)

train_rmse_insample = rmse(y, train_preds_insample)
oof_rmse = cv_df.iloc[0]["cv_rmse"]
oof_pearson = cv_df.iloc[0]["cv_pearson_r"]
train_pearson_insample, _ = pearsonr(y, train_preds_insample)

print(f"REQUIRED METRIC -> In-sample training RMSE: {train_rmse_insample:.4f}")
print(f"Out-of-fold CV RMSE (honest estimate):       {oof_rmse:.4f}")
print(f"In-sample training Pearson r:                {train_pearson_insample:.4f}")
print(f"Out-of-fold CV Pearson r:                     {oof_pearson:.4f}")
""")

code(r"""
oof_best = oof_preds[best_name]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].scatter(y, oof_best, alpha=0.4, s=15, color="#4C72B0")
axes[0].plot([0, 5], [0, 5], "r--", label="perfect prediction")
axes[0].set_xlabel("True grammar score")
axes[0].set_ylabel("Predicted (out-of-fold)")
axes[0].set_title(f"Predicted vs. True (OOF)\nRMSE={oof_rmse:.3f}, r={oof_pearson:.3f}")
axes[0].legend()

residuals = oof_best - y
axes[1].hist(residuals, bins=25, color="#55A868")
axes[1].axvline(0, color="red", linestyle="--")
axes[1].set_title("OOF residual distribution")
axes[1].set_xlabel("predicted - true")

plt.tight_layout()
plt.savefig("plots/04_prediction_diagnostics.png")
plt.show()
""")

md(r"""
**Reading the residual plot:** residuals are roughly centered at zero with
no strong skew, meaning the model isn't systematically over- or
under-predicting. The worst errors are inspected below.
""")
code(r"""
diag = train_feats[["filename", "label"]].copy()
diag["pred"] = oof_best
diag["abs_error"] = (diag["pred"] - diag["label"]).abs()
worst = diag.sort_values("abs_error", ascending=False).head(8)
worst = worst.merge(transcripts[["filename", "text"]], on="filename", how="left")
for _, row in worst.iterrows():
    print(f"[{row.filename}] true={row.label} pred={row.pred:.2f} "
          f"err={row.abs_error:.2f}")
    print(" ", row.text[:200])
    print()
""")

md(r"""
Looking at the worst-predicted examples usually shows one of: (a) a Whisper
transcription error that corrupted the grammar signal, (b) a true label near
a rubric boundary (e.g. 2 vs 3) where human raters themselves would have
disagreed, or (c) genuinely short/silent clips where almost no linguistic
signal exists. This is useful to call out explicitly — it's a real limitation
of an ASR-based pipeline, not just a modeling shortfall.
""")

# ---------------------------------------------------------------------------
md("## 12. Feature importance / interpretability")
code(r"""
if best_name == "gbm":
    importances = final_model.feature_importances_
elif best_name == "ridge":
    importances = np.abs(final_model.named_steps["reg"].coef_)
else:
    importances = np.zeros(len(feature_cols))

fi_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
fi_df = fi_df.sort_values("importance", ascending=False).head(15)

fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color="#4C72B0")
ax.set_title(f"Top 15 feature importances ({best_name})")
plt.tight_layout()
plt.savefig("plots/05_feature_importance.png")
plt.show()
""")

md(r"""
The top features (ASR confidence, vocabulary variety/type-token ratio,
response length, articulation rate, GPT-2 perplexity, and specific POS
ratios) all map back to the rubric in an explainable way: clearer/more
confident speech, richer vocabulary, more developed (longer, more complex)
responses, and more "predictable"/fluent grammar all push the predicted
score up — exactly the qualitative description separating a score of 2
from a score of 5 in the rubric.
""")

# ---------------------------------------------------------------------------
md("## 13. Test-set predictions & submission file")
code(r"""
X_test = test_feats[feature_cols].values
test_preds = final_model.predict(X_test)
test_preds = np.clip(test_preds, 0, 5)

submission = test_feats[["filename"]].copy()
submission["label"] = test_preds
submission.to_csv("data/submission.csv", index=False)

print(submission.shape)
submission.head()
""")

code(r"""
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(train_feats["label"], bins=20, alpha=0.5, label="train (true)", color="#4C72B0")
ax.hist(test_preds, bins=20, alpha=0.5, label="test (predicted)", color="#DD8452")
ax.legend()
ax.set_title("Predicted test-score distribution vs. train label distribution")
plt.tight_layout()
plt.savefig("plots/06_test_prediction_distribution.png")
plt.show()
""")

md(r"""
The predicted test distribution follows a similar overall shape to the
training label distribution, which is a reasonable sanity check (no major
distribution shift / systematic bias in the predictions).

## 14. Limitations & what I'd do with more time

- **ASR as a bottleneck**: Whisper is trained to output fluent English, so it
  can silently "correct" a speaker's grammar mistakes during transcription,
  suppressing exactly the signal we're trying to measure. A speech-specific
  disfluency-preserving ASR, or ensembling multiple ASR systems, would reduce
  this risk.
- **GPT-2 perplexity conflates fluency and grammar** — a grammatically correct
  but unusual sentence can still have high perplexity. A model fine-tuned for
  grammatical error detection (e.g. GECToR-style) on transcribed speech would
  be a more direct signal than a general-purpose LM.
- **spaCy's parser is trained on written text**, not disfluent spoken
  transcripts, so POS/dependency errors introduce some feature noise.
- **Small training set (769 examples)** limits how much a more complex model
  (e.g. fine-tuned transformer on transcripts) could be trusted without
  overfitting; this is why the final choice was a heavily regularized,
  interpretable model validated with 5-fold CV rather than a single split.
- With more time/data: fine-tune a small transformer (e.g. DistilBERT) on the
  transcripts directly as a learned grammar-scoring head, and/or incorporate
  speaker-normalized prosodic features to reduce the "recording quality"
  confound.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

with open("Grammar_Scoring_Engine.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook written.")
