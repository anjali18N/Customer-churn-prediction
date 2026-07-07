"""
generate_final_artifacts.py

Loads the trained model from 05_model_training.ipynb / 06_model_evaluation.ipynb,
reconstructs the same held-out test set, and produces a clean, standardized set of
final deliverables in results/:

    - best_model.pkl        (copy of the selected model)
    - metrics.csv           (final performance metrics at the tuned decision threshold)
    - confusion_matrix.png
    - roc_curve.png
    - feature_importance.png (permutation importance, not the biased impurity-based
                               importance — see 06_model_evaluation.ipynb for why)

Run from the project root:
    python src/generate_final_artifacts.py
"""

import json
import pickle
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
DATA_PATH = BASE_DIR / "data" / "processed" / "bank_customer_churn_features.csv"
MODEL_PATH = BASE_DIR / "models" / "trained_model.pkl"
EVAL_SUMMARY_PATH = BASE_DIR / "reports" / "model_evaluation_summary.json"
RESULTS_DIR = BASE_DIR / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_THRESHOLD_FALLBACK = 0.247  # from 06_model_evaluation.ipynb, used only if
                                     # reports/model_evaluation_summary.json is missing

sns.set_theme(style="whitegrid")


def load_chosen_threshold() -> float:
    """Read the tuned decision threshold from 06's saved report, if available."""
    if EVAL_SUMMARY_PATH.exists():
        with open(EVAL_SUMMARY_PATH) as f:
            summary = json.load(f)
        return float(summary["chosen_threshold"])
    print(
        f"Warning: {EVAL_SUMMARY_PATH} not found. "
        f"Falling back to hardcoded threshold {DEFAULT_THRESHOLD_FALLBACK}."
    )
    return DEFAULT_THRESHOLD_FALLBACK


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # --- Load model and reconstruct the exact test split used in training/evaluation ---
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"Loaded model: {type(model).__name__}")

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Test set: {X_test.shape[0]} rows, churn rate {y_test.mean():.2%}")

    threshold = load_chosen_threshold()
    print(f"Using decision threshold: {threshold:.3f}")

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # --- 1. best_model.pkl ---
    best_model_path = RESULTS_DIR / "best_model.pkl"
    shutil.copy(MODEL_PATH, best_model_path)
    print(f"Saved {best_model_path}")

    # --- 2. metrics.csv ---
    metrics = {
        "model": type(model).__name__,
        "decision_threshold": threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "average_precision": average_precision_score(y_test, y_proba),
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_path = RESULTS_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved {metrics_path}")
    print(metrics_df.to_string(index=False))

    # --- 3. confusion_matrix.png ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Retained", "Churned"]).plot(
        cmap="Blues", colorbar=False
    )
    plt.title(f"Confusion Matrix (threshold = {threshold:.3f})")
    plt.tight_layout()
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {cm_path}")

    # --- 4. roc_curve.png ---
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"{type(model).__name__} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    roc_path = RESULTS_DIR / "roc_curve.png"
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {roc_path}")

    # --- 5. feature_importance.png (permutation importance, not impurity-based) ---
    # Impurity-based importance was shown in 05/06 to be biased toward continuous
    # features (estimated_salary, credit_score) that aren't actually predictive.
    # Permutation importance is used here for a trustworthy, publishable result.
    perm_result = permutation_importance(
        model, X_test, y_test, scoring="roc_auc",
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": perm_result.importances_mean,
        "importance_std": perm_result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=importance_df, x="importance_mean", y="feature",
        xerr=importance_df["importance_std"]
    )
    plt.title("Permutation Importance (ROC-AUC decrease)")
    plt.xlabel("Mean importance")
    plt.tight_layout()
    fi_path = RESULTS_DIR / "feature_importance.png"
    plt.savefig(fi_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fi_path}")

    print("\nAll final artifacts generated in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
