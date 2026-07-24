"""
Churn Risk Scoring API

Serves the Random Forest model from Customer-churn-prediction as a real
service instead of a notebook artifact:
  POST /predict  -> churn probability, prediction, and risk tier
  POST /explain  -> same, plus per-customer SHAP explanation
  GET  /health   -> liveness check for the load balancer / cloud platform

Run locally:
    uvicorn app.main:app --reload

Docs (auto-generated):
    http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from app.model import explain, load_model, predict
from app.preprocessing import engineer_features
from app.schema import CustomerInput, ExplanationResponse, PredictionResponse

app = FastAPI(
    title="Bank Customer Churn Risk API",
    description="Real-time churn risk scoring with SHAP-based explanations, "
    "built on a Random Forest model tuned for churn recall.",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    # Load the model once when the container starts, not on every request
    load_model()

@app.get("/", include_in_schema=False)
def root():
    # So the bare URL doesn't 404 — sends visitors to the interactive API docs
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerInput):
    try:
        features_df = engineer_features(customer.dict())
        return predict(features_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/explain", response_model=ExplanationResponse)
def explain_churn(customer: CustomerInput):
    """Same as /predict, but also returns the top SHAP factors driving this
    specific customer's score — e.g. for a retention rep who needs to
    justify why a customer was flagged."""
    try:
        features_df = engineer_features(customer.dict())
        return explain(features_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
