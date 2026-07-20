"""
Loads the trained model once at startup and applies the TUNED decision
threshold — read dynamically from reports/model_evaluation_summary.json,
exactly like src/generate_final_artifacts.py does, with 0.247 only as a
fallback if that file isn't present.
"""

import json
import pickle
from pathlib import Path

import pandas as pd
import shap

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "trained_model.pkl"
EVAL_SUMMARY_PATH = BASE_DIR / "reports" / "model_evaluation_summary.json"
FALLBACK_THRESHOLD = 0.247  # from 06_model_evaluation.ipynb, used only if
                             # reports/model_evaluation_summary.json is missing

_model = None
_explainer = None
_threshold = None


def load_threshold() -> float:
    if EVAL_SUMMARY_PATH.exists():
        with open(EVAL_SUMMARY_PATH) as f:
            summary = json.load(f)
        return float(summary["chosen_threshold"])
    return FALLBACK_THRESHOLD


def load_model():
    """Loads the model and threshold once and caches them (called at API startup)."""
    global _model, _explainer, _threshold
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
    _threshold = load_threshold()
    # TreeExplainer works directly on Random Forest / XGBoost without a background dataset
    _explainer = shap.TreeExplainer(_model)
    return _model


def get_model():
    if _model is None:
        load_model()
    return _model


def get_explainer():
    if _explainer is None:
        load_model()
    return _explainer


def get_threshold() -> float:
    if _threshold is None:
        load_model()
    return _threshold


def risk_tier(probability: float) -> str:
    """Business-friendly tiers instead of a raw 0/1 flag or bare probability."""
    threshold = get_threshold()
    if probability < threshold:
        return "low"
    elif probability < 0.6:
        return "medium"
    else:
        return "high"


def predict(features_df: pd.DataFrame) -> dict:
    model = get_model()
    threshold = get_threshold()
    proba = float(model.predict_proba(features_df)[0][1])
    prediction = int(proba >= threshold)  # tuned threshold, NOT model.predict()
    return {
        "churn_probability": round(proba, 4),
        "churn_prediction": prediction,
        "risk_tier": risk_tier(proba),
        "threshold_used": threshold,
    }


def _extract_positive_class_row(shap_values):
    """Normalizes SHAP output across shap versions/explainer return shapes into
    a flat 1-D array of per-feature contributions for the positive (churn)
    class, for a single-row input. Handles:
      - list [class0_array, class1_array]                 (older shap)
      - ndarray shape (n_samples, n_features)              (binary, single output)
      - ndarray shape (n_samples, n_features, n_classes)   (newer shap, multi-class)
    """
    import numpy as np

    if isinstance(shap_values, list):
        # list of per-class arrays -> take class 1 (churn), row 0
        return shap_values[1][0]

    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        # (n_samples, n_features, n_classes) -> row 0, all features, class 1
        return arr[0, :, 1]
    elif arr.ndim == 2:
        # (n_samples, n_features) -> row 0
        return arr[0]
    else:
        raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")


def explain(features_df: pd.DataFrame, top_n: int = 5) -> dict:
    explainer = get_explainer()
    base = predict(features_df)

    shap_values = explainer.shap_values(features_df)
    values = _extract_positive_class_row(shap_values)

    contributions = list(zip(features_df.columns, values))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    top_factors = [
        {
            "feature": feat,
            "shap_value": round(float(val), 4),
            "direction": "increases risk" if val > 0 else "decreases risk",
        }
        for feat, val in contributions[:top_n]
    ]

    base["top_factors"] = top_factors
    return base
