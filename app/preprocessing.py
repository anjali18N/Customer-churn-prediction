"""
Recreates the feature engineering from 04_feature_engineering.ipynb so the API
can turn ONE raw customer record (from the request body) into the same feature
row your model was trained on.

IMPORTANT: This must stay in sync with your notebook. If you tweak feature
engineering in 04_feature_engineering.ipynb, update this file too — otherwise
the API and your training pipeline will silently drift apart, which is a
classic real-world source of training/serving skew.
"""

import pandas as pd

# Exact column list from your real feature set:
# df.drop(columns=["churn"]).columns.tolist()
MODEL_FEATURE_ORDER = [
    "credit_score",
    "gender",
    "age",
    "tenure",
    "balance",
    "credit_card",
    "active_member",
    "estimated_salary",
    "country_Germany",
    "country_Spain",
    "products_2",
    "products_3",
    "products_4",
    "is_zero_balance",
    "balance_to_salary_ratio",
    "age_group_31-45",
    "age_group_46-60",
    "age_group_60+",
    "tenure_by_age",
    "products_active_interaction",
]


def engineer_features(raw: dict) -> pd.DataFrame:
    """Takes a single raw customer dict (matching schema.CustomerInput) and
    returns a one-row DataFrame with engineered features in MODEL_FEATURE_ORDER.
    """
    row = dict(raw)  # shallow copy

    # --- gender: verified against your notebook —
    # df["gender"] = df["gender"].map({"Male": 0, "Female": 1})
    row["gender"] = 1 if row["gender"] == "Female" else 0

    # --- engineered numeric features (per README) ---
    row["is_zero_balance"] = 1 if row["balance"] == 0 else 0
    row["balance_to_salary_ratio"] = (
        row["balance"] / row["estimated_salary"] if row["estimated_salary"] > 0 else 0
    )
    row["tenure_by_age"] = row["tenure"] / row["age"] if row["age"] > 0 else 0
    row["products_active_interaction"] = row["products_number"] * row["active_member"]

    # --- age_group bins (exact column names from your real feature set) ---
    age = row["age"]
    row["age_group_31-45"] = 1 if 31 <= age <= 45 else 0
    row["age_group_46-60"] = 1 if 46 <= age <= 60 else 0
    row["age_group_60+"] = 1 if age > 60 else 0
    # age <= 30 is the implied reference category (all three flags = 0)

    # --- one-hot: country (France is the dropped reference category) ---
    row["country_Germany"] = 1 if row["country"] == "Germany" else 0
    row["country_Spain"] = 1 if row["country"] == "Spain" else 0

    # --- one-hot: products_number (products_1 is the dropped reference
    # category — matches the README's "2 products is the safe baseline"
    # finding, i.e. it makes sense for 1 product to be what everything else
    # is compared against) ---
    row["products_2"] = 1 if row["products_number"] == 2 else 0
    row["products_3"] = 1 if row["products_number"] == 3 else 0
    row["products_4"] = 1 if row["products_number"] == 4 else 0

    df = pd.DataFrame([row])
    return df[MODEL_FEATURE_ORDER]
