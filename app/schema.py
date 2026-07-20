"""
Request/response schemas for the churn prediction API.

Field names/types mirror the RAW columns from Bank Customer Churn Prediction.csv
(before any of your notebook feature engineering runs). Adjust field names here
if your raw column names differ slightly (e.g. "products_number" vs "num_products").
"""

from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    credit_score: int = Field(..., ge=300, le=900, example=650)
    country: str = Field(..., example="France")          # France / Germany / Spain
    gender: str = Field(..., example="Female")            # Male / Female
    age: int = Field(..., ge=18, le=100, example=42)
    tenure: int = Field(..., ge=0, le=15, example=3)
    balance: float = Field(..., ge=0, example=125000.50)
    products_number: int = Field(..., ge=1, le=4, example=2)
    credit_card: int = Field(..., ge=0, le=1, example=1)  # 1 = has credit card
    active_member: int = Field(..., ge=0, le=1, example=1)  # 1 = active
    estimated_salary: float = Field(..., ge=0, example=85000.00)

    class Config:
        json_schema_extra = {
            "example": {
                "credit_score": 650,
                "country": "Germany",
                "gender": "Female",
                "age": 52,
                "tenure": 2,
                "balance": 130000.0,
                "products_number": 1,
                "credit_card": 1,
                "active_member": 0,
                "estimated_salary": 78000.0,
            }
        }


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int          # 0 / 1, using the tuned 0.247 threshold
    risk_tier: str                 # low / medium / high
    threshold_used: float


class ExplanationResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_tier: str
    top_factors: list[dict]        # [{"feature": "...", "shap_value": ..., "direction": "..."}]
