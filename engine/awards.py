"""
CineMatch AI — Award-prediction supervised model.

Trains a binary classifier P(title is Emmy or Oscar nominated) on engineered
features from data/catalog.parquet and labels from data/awards_labeled.parquet.

Two algorithms are trained and compared with 5-fold stratified CV (ROC-AUC):
  - Logistic Regression (class-weight balanced)
  - Random Forest (class-weight balanced, n_estimators=200)

The model with the higher CV-AUC is persisted to data/award_model.pkl. Its
out-of-sample predictions on a held-out 20% test set are reported in
data/award_model_eval.json, and full-catalog probabilities are written to
data/award_predictions.parquet for the Streamlit UI to merge in.

Run:
    python3 -m engine.awards
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent
CATALOG_PATH = BASE / "data" / "catalog.parquet"
LABELS_PATH = BASE / "data" / "awards_labeled.parquet"
MODEL_PATH = BASE / "data" / "award_model.pkl"
PREDS_PATH = BASE / "data" / "award_predictions.parquet"
EVAL_PATH = BASE / "data" / "award_model_eval.json"


# ── Feature engineering ──────────────────────────────────────────────────────

def build_features(cat: pd.DataFrame) -> pd.DataFrame:
    """Engineer the feature matrix used by the classifier.

    Inputs come from the cleaned catalog (already z-scored on rating / votes /
    year / popularity). We add a multi-hot genre encoding and a one-hot decade
    encoding so the model can pick up era and genre effects.
    """
    df = pd.DataFrame(index=cat.index)
    df["rating_z"]      = cat["rating_z"].fillna(0.0)
    df["votes_z"]       = cat["votes_z"].fillna(0.0)
    df["start_year_z"]  = cat["start_year_z"].fillna(0.0)
    df["popularity_z"]  = cat["popularity_z"].fillna(0.0)

    # votes_log_z — higher entropy than raw votes_z (already z'd but on raw)
    votes = cat["votes"].fillna(0.0).clip(lower=0.0)
    votes_log = np.log1p(votes)
    mu, sd = votes_log.mean(), votes_log.std() or 1.0
    df["votes_log_z"] = (votes_log - mu) / sd

    # decade one-hot
    decades = cat["decade_str"].fillna("unknown")
    for d in sorted(decades.unique()):
        df[f"decade_{d}"] = (decades == d).astype(int)

    # genre multi-hot (keep top-15 by frequency to bound dimensionality)
    genre_lists = cat["genres"].fillna("").str.split(", ")
    flat = pd.Series([g for sub in genre_lists for g in sub if g])
    top_genres = flat.value_counts().head(15).index.tolist()
    for g in top_genres:
        df[f"genre_{g}"] = genre_lists.apply(lambda lst: int(g in lst))

    return df


# ── Training ─────────────────────────────────────────────────────────────────

def _precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    order = np.argsort(-y_prob)
    top_k = order[:k]
    return float(np.mean(y_true[top_k]))


def _recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    pos = y_true.sum()
    if pos == 0:
        return 0.0
    order = np.argsort(-y_prob)
    top_k = order[:k]
    return float(np.sum(y_true[top_k]) / pos)


def train() -> None:
    print("→ Loading catalog + labels…")
    cat = pd.read_parquet(CATALOG_PATH)
    labels = pd.read_parquet(LABELS_PATH)
    cat = cat.merge(labels[["title", "emmy_nominated", "oscar_nominated"]],
                    on="title", how="left").fillna({"emmy_nominated": 0,
                                                    "oscar_nominated": 0})
    cat["y"] = ((cat["emmy_nominated"] == 1) | (cat["oscar_nominated"] == 1)).astype(int)
    n_pos = int(cat["y"].sum())
    print(f"   {n_pos}/{len(cat)} positive ({n_pos / len(cat):.1%})")

    print("→ Building features…")
    X = build_features(cat)
    y = cat["y"].values
    feature_cols = list(X.columns)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    X_tr, X_te, y_tr, y_te = train_test_split(
        Xs, y, test_size=0.20, stratify=y, random_state=42
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    eval_report: dict = {"models": {}, "feature_cols": feature_cols,
                         "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
                         "n_positive_train": int(y_tr.sum()),
                         "n_positive_test": int(y_te.sum())}

    best_name, best_auc, best_model = None, -1.0, None
    for name, mdl in models.items():
        print(f"→ 5-fold CV on {name}…")
        cv_aucs = []
        for tr_idx, va_idx in cv.split(X_tr, y_tr):
            mdl.fit(X_tr[tr_idx], y_tr[tr_idx])
            p = mdl.predict_proba(X_tr[va_idx])[:, 1]
            cv_aucs.append(roc_auc_score(y_tr[va_idx], p))
        cv_auc_mean = float(np.mean(cv_aucs))
        cv_auc_std = float(np.std(cv_aucs))

        mdl.fit(X_tr, y_tr)
        p_te = mdl.predict_proba(X_te)[:, 1]
        test_auc = float(roc_auc_score(y_te, p_te))
        test_ap = float(average_precision_score(y_te, p_te))
        prec10 = _precision_at_k(y_te, p_te, 10)
        rec10 = _recall_at_k(y_te, p_te, 10)
        prec50 = _precision_at_k(y_te, p_te, 50)
        cm = confusion_matrix(y_te, (p_te >= 0.5).astype(int)).tolist()

        eval_report["models"][name] = {
            "cv_auc_mean": cv_auc_mean,
            "cv_auc_std": cv_auc_std,
            "cv_auc_folds": [float(x) for x in cv_aucs],
            "test_roc_auc": test_auc,
            "test_avg_precision": test_ap,
            "precision_at_10": prec10,
            "recall_at_10": rec10,
            "precision_at_50": prec50,
            "confusion_matrix_at_0.5": cm,
        }
        print(f"   CV-AUC = {cv_auc_mean:.3f} ± {cv_auc_std:.3f}  ·  "
              f"test AUC = {test_auc:.3f}  ·  P@10 = {prec10:.2f}")

        if cv_auc_mean > best_auc:
            best_auc = cv_auc_mean
            best_name = name
            best_model = mdl

    eval_report["best_model"] = best_name
    eval_report["best_cv_auc"] = best_auc

    # Retrain best on full data and emit catalog-wide probabilities
    print(f"→ Re-fitting best model ({best_name}) on full data…")
    best_model.fit(Xs, y)
    all_probs = best_model.predict_proba(Xs)[:, 1]

    preds = pd.DataFrame({
        "title": cat["title"].values,
        "award_probability": all_probs.round(4),
    })
    preds.to_parquet(PREDS_PATH, index=False)

    # Bundle scaler + model so the pickle is self-contained
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": best_model, "scaler": scaler,
                     "feature_cols": feature_cols}, f)

    EVAL_PATH.write_text(json.dumps(eval_report, indent=2))

    print(f"✅ Wrote {PREDS_PATH}, {MODEL_PATH}, {EVAL_PATH}")
    print(f"   Top-10 predicted award contenders:")
    top = preds.sort_values("award_probability", ascending=False).head(10)
    for _, r in top.iterrows():
        print(f"     {r['award_probability']:.2f}  {r['title']}")


if __name__ == "__main__":
    train()
