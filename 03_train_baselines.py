# 03_train_baselines.py
# Trains 4 baseline models (NB, LR, RF, XGBoost) on each dataset split.
# SMOTE applied on training set only after TF-IDF transformation.
# Saves results to baseline_results.csv, convergence data to convergence/,
# and ROC curve data to roc_data/ (added: hybrid model dropped from paper
# scope, so ROC curves for these 4 baselines are now the headline comparison).

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             brier_score_loss, log_loss,
                             roc_curve, roc_auc_score)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# ── Paths ────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
RESULTS_OUT = os.path.join(BASE, "baseline_results.csv")
CONV_DIR    = os.path.join(BASE, "convergence")
ROC_DIR     = os.path.join(BASE, "roc_data")           # NEW
os.makedirs(CONV_DIR, exist_ok=True)
os.makedirs(ROC_DIR, exist_ok=True)                    # NEW

DATASETS = {
    "SpamAssassin" : os.path.join(BASE, "spamassassin_cleaned.csv"),
    "SMS"          : os.path.join(BASE, "sms_cleaned.csv"),
    "Pooled"       : os.path.join(BASE, "pooled_cleaned.csv"),
}

RANDOM_STATE = 42
TEST_SIZE    = 0.20
MAX_FEATURES = 3000


def load_dataset(path):
    df = pd.read_csv(path, dtype_backend="numpy_nullable")
    df = df.dropna(subset=["label", "text"])
    df["text"]  = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    df = df.copy()
    return df


def apply_smote(X_train, y_train, corpus_name):
    """
    Applies SMOTE to the training set only.
    SMOTE generates synthetic spam samples to balance the class distribution.
    Test set is never touched.
    """
    before_counts = np.bincount(y_train)
    print(f"  [SMOTE] Before: Ham={before_counts[0]}  Spam={before_counts[1]}")

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    after_counts = np.bincount(y_resampled)
    print(f"  [SMOTE] After : Ham={after_counts[0]}  Spam={after_counts[1]}")
    print(f"  [SMOTE] Synthetic spam samples added: "
          f"{after_counts[1] - before_counts[1]}")

    return X_resampled, y_resampled


# ── NEW: ROC curve data capture ──────────────────────────────────────────────
def save_roc_data(model_key, y_true, y_prob, corpus_tag):
    """
    Computes and saves (fpr, tpr, auc) for one model on one corpus's test set.
    model_key should be one of: 'nb', 'lr', 'rf', 'xgb' — matches the naming
    convention already used for the convergence JSON files.
    Saved as .npz so plot_results.py can load fpr/tpr/auc directly without
    retraining anything.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    out_path = os.path.join(ROC_DIR, f"{corpus_tag}_{model_key}_roc.npz")
    np.savez(out_path, fpr=fpr, tpr=tpr, auc=auc)
    return auc


def compute_metrics(name, y_true, y_pred, y_prob):
    return {
        "model"     : name,
        "accuracy"  : round(accuracy_score(y_true, y_pred),       4),
        "precision" : round(precision_score(y_true, y_pred,
                                            zero_division=0),      4),
        "recall"    : round(recall_score(y_true, y_pred,
                                         zero_division=0),         4),
        "f1"        : round(f1_score(y_true, y_pred,
                                     zero_division=0),             4),
        "brier"     : round(brier_score_loss(y_true, y_prob),      4),
        "log_loss"  : round(log_loss(y_true, y_prob),              4),
        "roc_auc"   : round(roc_auc_score(y_true, y_prob),         4),  # NEW
    }


def train_nb(X_train, X_test, y_train, y_test, corpus):
    print("  [NB] Training Multinomial Naive Bayes...")

    # NOTE: SMOTE may produce negative values in dense arrays.
    # MinMaxScaler ensures non-negative input for MultinomialNB.
    from sklearn.preprocessing import MinMaxScaler
    import scipy.sparse as sp

    if sp.issparse(X_train):
        X_train_dense = X_train.toarray()
        X_test_dense  = X_test.toarray()
    else:
        X_train_dense = np.array(X_train)
        X_test_dense  = np.array(X_test)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_dense)
    X_test_scaled  = scaler.transform(X_test_dense)

    model = MultinomialNB()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    conv = {"note": "Closed-form: no iterative training curve."}
    with open(os.path.join(CONV_DIR,
              f"{corpus}_nb_convergence.json"), "w") as f:
        json.dump(conv, f)

    auc = save_roc_data("nb", y_test, y_prob, corpus)          # NEW

    metrics = compute_metrics("Naive Bayes", y_test, y_pred, y_prob)
    print(f"     Accuracy={metrics['accuracy']}  F1={metrics['f1']}  "
          f"AUC={auc:.4f}")
    return metrics, model


def train_lr(X_train, X_test, y_train, y_test, corpus):
    print("  [LR] Training Logistic Regression...")
    model = LogisticRegression(
        solver="saga",
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    conv = {
        "iterations_run" : int(model.n_iter_[0]),
        "note"           : "SAGA solver; n_iter_ = iterations until convergence."
    }
    with open(os.path.join(CONV_DIR,
              f"{corpus}_lr_convergence.json"), "w") as f:
        json.dump(conv, f)

    auc = save_roc_data("lr", y_test, y_prob, corpus)          # NEW

    metrics = compute_metrics("Logistic Regression", y_test, y_pred, y_prob)
    print(f"     Accuracy={metrics['accuracy']}  F1={metrics['f1']}  "
          f"Iterations={model.n_iter_[0]}  AUC={auc:.4f}")
    return metrics, model


def train_rf(X_train, X_test, y_train, y_test, corpus):
    print("  [RF] Training Random Forest...")
    n_estimators = 200
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        oob_score=True,
        warm_start=True
    )

    oob_errors = []
    step = 10
    for n in range(step, n_estimators + 1, step):
        model.n_estimators = n
        model.fit(X_train, y_train)
        oob_errors.append({
            "n_trees"  : n,
            "oob_error": round(1 - model.oob_score_, 4)
        })
        print(f"     Trees={n:3d}  OOB error={oob_errors[-1]['oob_error']:.4f}")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    conv = {"oob_curve": oob_errors}
    with open(os.path.join(CONV_DIR,
              f"{corpus}_rf_convergence.json"), "w") as f:
        json.dump(conv, f)

    auc = save_roc_data("rf", y_test, y_prob, corpus)          # NEW

    metrics = compute_metrics("Random Forest", y_test, y_pred, y_prob)
    print(f"     Accuracy={metrics['accuracy']}  F1={metrics['f1']}  "
          f"AUC={auc:.4f}")
    return metrics, model


def train_xgb(X_train, X_test, y_train, y_test, corpus):
    print("  [XGB] Training XGBoost with early stopping...")
    import scipy.sparse as sp

    if sp.issparse(X_train):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest  = xgb.DMatrix(X_test,  label=y_test)
    else:
        dtrain = xgb.DMatrix(np.array(X_train), label=y_train)
        dtest  = xgb.DMatrix(np.array(X_test),  label=y_test)

    params = {
        "objective"        : "binary:logistic",
        "eval_metric"      : "logloss",
        "eta"              : 0.1,
        "max_depth"        : 6,
        "subsample"        : 0.8,
        "colsample_bytree" : 0.8,
        "seed"             : RANDOM_STATE,
        "nthread"          : -1,
    }

    evals_result = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        evals=[(dtrain, "train"), (dtest, "val")],
        early_stopping_rounds=20,
        evals_result=evals_result,
        verbose_eval=10,
    )

    conv = {
        "train_logloss" : evals_result["train"]["logloss"],
        "val_logloss"   : evals_result["val"]["logloss"],
        "best_round"    : int(booster.best_iteration),
    }
    with open(os.path.join(CONV_DIR,
              f"{corpus}_xgb_convergence.json"), "w") as f:
        json.dump(conv, f)

    y_prob = booster.predict(dtest)
    y_pred = (y_prob >= 0.5).astype(int)

    auc = save_roc_data("xgb", y_test, y_prob, corpus)         # NEW

    metrics = compute_metrics("XGBoost", y_test, y_pred, y_prob)
    print(f"     Accuracy={metrics['accuracy']}  F1={metrics['f1']}  "
          f"Best round={booster.best_iteration}  AUC={auc:.4f}")
    return metrics, booster


def run_corpus(corpus_name, path):
    print(f"\n{'='*60}")
    print(f" Dataset: {corpus_name}")
    print(f"{'='*60}")

    df = load_dataset(path)
    print(f"  Loaded {len(df)} rows | "
          f"spam={df['label'].sum()}  "
          f"ham={(df['label']==0).sum()}")

    X_raw = np.array(df["text"].tolist(), dtype=object)
    y     = np.array(df["label"].tolist(), dtype=int)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y
    )

    print(f"  Train={len(y_train)}  Test={len(y_test)}")

    # TF-IDF vectorisation
    tfidf = TfidfVectorizer(
        max_features=MAX_FEATURES,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-zA-Z]{2,}\b",
        ngram_range=(1, 2),
    )
    X_train = tfidf.fit_transform(X_train_raw)
    X_test  = tfidf.transform(X_test_raw)

    # Apply SMOTE to training set only
    print()
    X_train_sm, y_train_sm = apply_smote(X_train, y_train, corpus_name)
    print()

    all_metrics = []
    tag = corpus_name.replace(" ", "_")

    m, _ = train_nb(X_train_sm, X_test, y_train_sm, y_test, tag)
    m["corpus"] = corpus_name
    all_metrics.append(m)

    m, _ = train_lr(X_train_sm, X_test, y_train_sm, y_test, tag)
    m["corpus"] = corpus_name
    all_metrics.append(m)

    m, _ = train_rf(X_train_sm, X_test, y_train_sm, y_test, tag)
    m["corpus"] = corpus_name
    all_metrics.append(m)

    m, _ = train_xgb(X_train_sm, X_test, y_train_sm, y_test, tag)
    m["corpus"] = corpus_name
    all_metrics.append(m)

    return all_metrics


if __name__ == "__main__":
    all_results = []

    for name, path in DATASETS.items():
        results = run_corpus(name, path)
        all_results.extend(results)

    df_results = pd.DataFrame(all_results)
    cols = ["corpus", "model", "accuracy", "precision",
            "recall", "f1", "brier", "log_loss", "roc_auc"]   # roc_auc added
    df_results = df_results[cols]
    df_results.to_csv(RESULTS_OUT, index=False)

    print(f"\n{'='*60}")
    print("BASELINE RESULTS SUMMARY")
    print('='*60)
    print(df_results.to_string(index=False))
    print(f"\nSaved to {RESULTS_OUT}")
    print(f"ROC curve data saved to {ROC_DIR}")