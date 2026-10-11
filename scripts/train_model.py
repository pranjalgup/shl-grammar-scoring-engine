"""Merge features with labels, run CV model comparison, fit final model,
generate test predictions and diagnostic plots.

Outputs:
  data/model_cv_results.csv       - per-model CV RMSE/Pearson summary
  data/oof_predictions.csv        - out-of-fold train predictions (for plots)
  data/submission.csv             - final test predictions
  data/feature_importance.csv     - feature importances of the chosen model
  plots/*.png                     - diagnostic visualizations
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
PLOTS = os.path.join(ROOT, "plots")
os.makedirs(PLOTS, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

ID_COLS = {"split", "filename"}


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def load_data():
    feats = pd.read_csv(os.path.join(DATA, "Dataset_Final", "..", "features.csv"))
    train_labels = pd.read_csv(os.path.join(DATA, "Dataset_Final", "train.csv"))
    test_labels = pd.read_csv(os.path.join(DATA, "Dataset_Final", "test.csv"))

    train_feats = feats[feats["split"] == "train"].merge(
        train_labels, on="filename", how="inner"
    )
    test_feats = feats[feats["split"] == "test"].merge(
        test_labels[["filename"]], on="filename", how="inner"
    )
    assert len(train_feats) == len(train_labels), "train merge mismatch"
    assert len(test_feats) == len(test_labels), "test merge mismatch"

    feature_cols = [c for c in feats.columns if c not in ID_COLS]
    return train_feats, test_feats, feature_cols


def build_models():
    return {
        "ridge": Pipeline(
            [("scaler", StandardScaler()), ("reg", Ridge(alpha=5.0, random_state=RANDOM_STATE))]
        ),
        "svr_rbf": Pipeline(
            [("scaler", StandardScaler()), ("reg", SVR(kernel="rbf", C=3.0, epsilon=0.2))]
        ),
        "gbm": GradientBoostingRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
    }


def cross_validate(train_feats, feature_cols):
    X = train_feats[feature_cols].values
    y = train_feats["label"].values
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    results = []
    oof_preds = {}
    for name, model in build_models().items():
        oof = np.zeros(len(y))
        for train_idx, val_idx in kf.split(X):
            m = build_models()[name]  # fresh clone each fold
            m.fit(X[train_idx], y[train_idx])
            oof[val_idx] = m.predict(X[val_idx])
        r = rmse(y, oof)
        corr, _ = pearsonr(y, oof)
        results.append({"model": name, "cv_rmse": r, "cv_pearson": corr})
        oof_preds[name] = oof
        print(f"{name}: RMSE={r:.4f}  Pearson r={corr:.4f}", flush=True)

    return pd.DataFrame(results), oof_preds


def main():
    train_feats, test_feats, feature_cols = load_data()
    print(f"Train: {train_feats.shape}, Test: {test_feats.shape}, "
          f"n_features={len(feature_cols)}", flush=True)

    cv_df, oof_preds = cross_validate(train_feats, feature_cols)
    cv_df.to_csv(os.path.join(DATA, "model_cv_results.csv"), index=False)

    best_name = cv_df.sort_values("cv_rmse").iloc[0]["model"]
    print(f"\nBest model by CV RMSE: {best_name}", flush=True)

    oof_df = train_feats[["filename", "label"]].copy()
    for name, preds in oof_preds.items():
        oof_df[f"pred_{name}"] = preds
    oof_df.to_csv(os.path.join(DATA, "oof_predictions.csv"), index=False)

    # Fit chosen model on full training data, predict test set
    final_model = build_models()[best_name]
    X_train = train_feats[feature_cols].values
    y_train = train_feats["label"].values
    final_model.fit(X_train, y_train)

    X_test = test_feats[feature_cols].values
    test_preds = final_model.predict(X_test)
    test_preds = np.clip(test_preds, 0, 5)

    submission = test_feats[["filename"]].copy()
    submission["label"] = test_preds
    submission.to_csv(os.path.join(DATA, "submission.csv"), index=False)

    # feature importance / coefficients
    if best_name == "gbm":
        importances = final_model.feature_importances_
    elif best_name == "ridge":
        importances = np.abs(final_model.named_steps["reg"].coef_)
    else:
        importances = np.zeros(len(feature_cols))  # SVR has no simple importances

    fi_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False)
    fi_df.to_csv(os.path.join(DATA, "feature_importance.csv"), index=False)

    train_rmse_final = rmse(y_train, final_model.predict(X_train))
    print(f"\nFinal model ({best_name}) in-sample train RMSE: {train_rmse_final:.4f}")
    print(f"(Out-of-fold CV RMSE, the honest estimate: "
          f"{cv_df[cv_df.model==best_name].cv_rmse.values[0]:.4f})")
    print("\nTop 10 features:")
    print(fi_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
