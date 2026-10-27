# 05_plot_results.py
# Generates all figures for the paper:
#   1. Metric comparison LINE charts (4 baselines x 3 corpora)   [was bar charts]
#   2. Convergence curves (LR iterations, RF OOB error, XGBoost log-loss)
#   3. ROC curves for the 4 baselines, one figure per corpus     [NEW]
#
# NOTE: Hybrid model dropped from paper scope per advisor feedback.
# The old Figure 3 (Hybrid vs XGBoost calibration) has been removed —
# hybrid_results.csv is no longer read or plotted.

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Paths ────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
CONV_DIR    = os.path.join(BASE, "convergence")
ROC_DIR     = os.path.join(BASE, "roc_data")           # NEW
FIGS_DIR    = os.path.join(BASE, "figures")
os.makedirs(FIGS_DIR, exist_ok=True)

BASELINE_CSV = os.path.join(BASE, "baseline_results.csv")
# HYBRID_CSV removed — hybrid model out of scope

MODEL_COLORS = {
    "Naive Bayes"         : "#4C72B0",
    "Logistic Regression" : "#DD8452",
    "Random Forest"       : "#55A868",
    "XGBoost"             : "#C44E52",
}

MODEL_KEY_TO_LABEL = {
    "nb"  : "Naive Bayes",
    "lr"  : "Logistic Regression",
    "rf"  : "Random Forest",
    "xgb" : "XGBoost",
}

CORPORA = ["SpamAssassin", "SMS", "Pooled"]


# ── Helper ───────────────────────────────────────────────────────────────────
def save(fig, name):
    path = os.path.join(FIGS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ── Figure 1: Baseline metric comparison — LINE charts ───────────────────────
# Changed from grouped bar charts to line charts per requested format:
# x-axis = corpus, y-axis = metric value, one line per model.
def plot_baseline_metrics():
    df = pd.read_csv(BASELINE_CSV)
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

    models_in_order = ["Naive Bayes", "Logistic Regression",
                       "Random Forest", "XGBoost"]

    for metric, label in zip(metrics, metric_labels):
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.suptitle(f"Baseline Comparison — {label}",
                     fontsize=13, fontweight="bold")

        for model in models_in_order:
            sub = df[df["model"] == model].set_index("corpus")
            sub = sub.reindex(CORPORA)  # ensure consistent x-order
            ax.plot(CORPORA, sub[metric], marker="o", linewidth=2,
                   label=model, color=MODEL_COLORS.get(model, "#999"))
            for x, y in zip(CORPORA, sub[metric]):
                if pd.notna(y):
                    ax.annotate(f"{y:.3f}", (x, y),
                               textcoords="offset points", xytext=(0, 6),
                               ha="center", fontsize=7)

        ax.set_xlabel("Dataset")
        ax.set_ylabel(label)
        ax.set_ylim(0.5, 1.02)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        save(fig, f"baseline_{metric}.png")


# ── Figure 2: Convergence curves ─────────────────────────────────────────────
def plot_convergence():
    for corpus_tag in ["SpamAssassin", "SMS", "Pooled"]:
        tag = corpus_tag.replace(" ", "_")

        # XGBoost log-loss curve
        xgb_path = os.path.join(CONV_DIR, f"{tag}_xgb_convergence.json")
        if os.path.exists(xgb_path):
            with open(xgb_path) as f:
                data = json.load(f)
            train_ll = data["train_logloss"]
            val_ll   = data["val_logloss"]
            rounds   = list(range(1, len(train_ll) + 1))

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(rounds, train_ll, label="Train log-loss", color="#C44E52")
            ax.plot(rounds, val_ll,   label="Val log-loss",
                    color="#C44E52", linestyle="--")
            best = data.get("best_round", len(rounds))
            ax.axvline(best, color="gray", linestyle=":", label=f"Best round ({best})")
            ax.set_xlabel("Boosting Round")
            ax.set_ylabel("Log-Loss")
            ax.set_title(f"XGBoost Convergence — {corpus_tag}")
            ax.legend()
            plt.tight_layout()
            save(fig, f"convergence_xgb_{tag}.png")

        # RF OOB error curve
        rf_path = os.path.join(CONV_DIR, f"{tag}_rf_convergence.json")
        if os.path.exists(rf_path):
            with open(rf_path) as f:
                data = json.load(f)
            oob = data["oob_curve"]
            trees  = [d["n_trees"]   for d in oob]
            errors = [d["oob_error"] for d in oob]

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(trees, errors, color="#55A868", marker="o", markersize=3)
            ax.set_xlabel("Number of Trees")
            ax.set_ylabel("OOB Error")
            ax.set_title(f"Random Forest Convergence — {corpus_tag}")
            plt.tight_layout()
            save(fig, f"convergence_rf_{tag}.png")

        # LR iterations
        lr_path = os.path.join(CONV_DIR, f"{tag}_lr_convergence.json")
        if os.path.exists(lr_path):
            with open(lr_path) as f:
                data = json.load(f)
            print(f"  LR ({corpus_tag}): converged in "
                  f"{data.get('iterations_run', 'N/A')} iterations.")


# ── Figure 3: ROC curves — NEW, replaces hybrid comparison plot ─────────────
def plot_roc_curves():
    """
    One ROC figure per corpus, with all 4 baseline models plotted together
    (each line loaded from the .npz files saved by 03_train_baselines.py).
    """
    for corpus in CORPORA:
        tag = corpus.replace(" ", "_")
        fig, ax = plt.subplots(figsize=(6, 6))

        any_found = False
        for key, label in MODEL_KEY_TO_LABEL.items():
            roc_path = os.path.join(ROC_DIR, f"{tag}_{key}_roc.npz")
            if not os.path.exists(roc_path):
                print(f"  [ROC] Missing: {roc_path} — skipping {label} for {corpus}")
                continue
            any_found = True
            data = np.load(roc_path)
            fpr, tpr, auc = data["fpr"], data["tpr"], float(data["auc"])
            ax.plot(fpr, tpr, linewidth=2,
                   color=MODEL_COLORS.get(label, "#999"),
                   label=f"{label} (AUC = {auc:.3f})")

        if not any_found:
            print(f"  [ROC] No ROC data found for {corpus} — skipping figure.")
            plt.close(fig)
            continue

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray",
               linewidth=1, label="Chance (AUC = 0.500)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves — {corpus}", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)

        plt.tight_layout()
        save(fig, f"roc_{tag}.png")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Plotting baseline metric comparisons (line charts)...")
    plot_baseline_metrics()

    print("\nPlotting convergence curves...")
    plot_convergence()

    print("\nPlotting ROC curves...")
    plot_roc_curves()

    print(f"\nAll figures saved to: {FIGS_DIR}")