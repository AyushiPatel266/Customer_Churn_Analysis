from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import os

# Initialize app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts churn probability and retention strategy for telecom customers",
    version="1.0.0"
)

# Allow Flask frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
MODEL_PATH = "models/xgb_churn_model.pkl"
ENCODER_PATH = "models/label_encoders.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully")


# Define input schema
class CustomerInput(BaseModel):
    gender: str                    # Male / Female
    SeniorCitizen: int             # 0 or 1
    Partner: int                   # 0 or 1
    Dependents: int                # 0 or 1
    tenure: int                    # months
    PhoneService: int              # 0 or 1
    MultipleLines: str             # Yes / No / No phone service
    InternetService: str           # DSL / Fiber optic / No
    OnlineSecurity: str            # Yes / No / No internet service
    OnlineBackup: str              # Yes / No / No internet service
    DeviceProtection: str          # Yes / No / No internet service
    TechSupport: str               # Yes / No / No internet service
    StreamingTV: str               # Yes / No / No internet service
    StreamingMovies: str           # Yes / No / No internet service
    Contract: str                  # Month-to-month / One year / Two year
    PaperlessBilling: int          # 0 or 1
    PaymentMethod: str             # Electronic check / Mailed check / etc
    MonthlyCharges: float          # e.g. 65.5
    TotalCharges: float            # e.g. 1200.0


def preprocess_input(data: CustomerInput) -> pd.DataFrame:
    """Convert input to model-ready dataframe"""

    # Hardcoded mappings
    encoding_maps = {
        'gender': {'Female': 0, 'Male': 1},
        'MultipleLines': {'No': 0, 'No phone service': 1, 'Yes': 2},
        'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
        'OnlineSecurity': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'OnlineBackup': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'DeviceProtection': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'TechSupport': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'StreamingTV': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'StreamingMovies': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
        'PaymentMethod': {
            'Bank transfer (automatic)': 0,
            'Credit card (automatic)': 1,
            'Electronic check': 2,
            'Mailed check': 3
        }
    }

    # Step 1: Binary engineered features using original string values
    ml = data.MultipleLines
    is_val = data.InternetService

    multilines_bin = 1 if ml == 'Yes' else 0
    internet_bin = 1 if is_val in ['Fiber optic', 'DSL'] else 0
    security_bin = 1 if data.OnlineSecurity == 'Yes' else 0
    backup_bin = 1 if data.OnlineBackup == 'Yes' else 0
    device_bin = 1 if data.DeviceProtection == 'Yes' else 0
    tech_bin = 1 if data.TechSupport == 'Yes' else 0
    tv_bin = 1 if data.StreamingTV == 'Yes' else 0
    movies_bin = 1 if data.StreamingMovies == 'Yes' else 0

    num_services = (data.PhoneService + multilines_bin + internet_bin +
                    security_bin + backup_bin + device_bin +
                    tech_bin + tv_bin + movies_bin)

    # Step 2: Encode categorical columns using hardcoded maps
    gender_enc = encoding_maps['gender'].get(data.gender, 0)
    multilines_enc = encoding_maps['MultipleLines'].get(data.MultipleLines, 0)
    internet_enc = encoding_maps['InternetService'].get(data.InternetService, 0)
    security_enc = encoding_maps['OnlineSecurity'].get(data.OnlineSecurity, 0)
    backup_enc = encoding_maps['OnlineBackup'].get(data.OnlineBackup, 0)
    device_enc = encoding_maps['DeviceProtection'].get(data.DeviceProtection, 0)
    tech_enc = encoding_maps['TechSupport'].get(data.TechSupport, 0)
    tv_enc = encoding_maps['StreamingTV'].get(data.StreamingTV, 0)
    movies_enc = encoding_maps['StreamingMovies'].get(data.StreamingMovies, 0)
    contract_enc = encoding_maps['Contract'].get(data.Contract, 0)
    payment_enc = encoding_maps['PaymentMethod'].get(data.PaymentMethod, 0)

    # Step 3: Build final dataframe in exact training order
    final_row = {
        'gender': gender_enc,
        'SeniorCitizen': data.SeniorCitizen,
        'Partner': data.Partner,
        'Dependents': data.Dependents,
        'tenure': data.tenure,
        'PhoneService': data.PhoneService,
        'MultipleLines': multilines_enc,
        'InternetService': internet_enc,
        'OnlineSecurity': security_enc,
        'OnlineBackup': backup_enc,
        'DeviceProtection': device_enc,
        'TechSupport': tech_enc,
        'StreamingTV': tv_enc,
        'StreamingMovies': movies_enc,
        'Contract': contract_enc,
        'PaperlessBilling': data.PaperlessBilling,
        'PaymentMethod': payment_enc,
        'MonthlyCharges': data.MonthlyCharges,
        'TotalCharges': data.TotalCharges,
        'MultipleLines_bin': multilines_bin,
        'InternetService_bin': internet_bin,
        'OnlineSecurity_bin': security_bin,
        'OnlineBackup_bin': backup_bin,
        'DeviceProtection_bin': device_bin,
        'TechSupport_bin': tech_bin,
        'StreamingTV_bin': tv_bin,
        'StreamingMovies_bin': movies_bin,
        'num_services': num_services
    }

    return pd.DataFrame([final_row])

def get_strategy(churn_prob: float, monthly_charges: float, tenure: int) -> dict:
    """Generate retention strategy based on prediction"""
    
    clv_proxy = monthly_charges * max(tenure, 1)
    
    # Thresholds based on dataset medians
    high_risk = churn_prob >= 0.5
    high_value = clv_proxy >= 1500

    if high_risk and high_value:
        quadrant = "Priority: Retain Now"
        action = "Immediate personal outreach. Offer contract upgrade and discount."
        priority = "HIGH"
        budget = "High retention spend justified"
    elif not high_risk and high_value:
        quadrant = "Protect and Reward"
        action = "Enroll in loyalty program. Offer early access to new features."
        priority = "MEDIUM"
        budget = "Moderate spend on rewards and engagement"
    elif high_risk and not high_value:
        quadrant = "Do Not Invest"
        action = "No retention spend. Let churn naturally."
        priority = "LOW"
        budget = "Zero retention spend recommended"
    else:
        quadrant = "Low Priority"
        action = "Standard service. Low-cost email engagement only."
        priority = "LOW"
        budget = "Minimal spend"

    return {
        "quadrant": quadrant,
        "action": action,
        "priority": priority,
        "budget_recommendation": budget
    }


# Routes

@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict_churn(customer: CustomerInput):
    try:
        # Preprocess input
        df = preprocess_input(customer)
        
        # Get prediction
        churn_prob = float(model.predict_proba(df)[:, 1][0])
        churn_prediction = int(churn_prob >= 0.5)
        
        # Calculate CLV
        projected_clv = round(customer.MonthlyCharges * 12 * (1 - churn_prob), 2)
        
        # Get risk level
        if churn_prob >= 0.7:
            risk_level = "Critical"
        elif churn_prob >= 0.5:
            risk_level = "High"
        elif churn_prob >= 0.3:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Get strategy
        strategy = get_strategy(churn_prob, customer.MonthlyCharges, customer.tenure)
        
        return {
            "churn_probability": round(churn_prob * 100, 1),
            "churn_prediction": "Will Churn" if churn_prediction == 1 else "Will Stay",
            "risk_level": risk_level,
            "projected_12m_clv": projected_clv,
            "strategy": strategy
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/predict/batch")
def predict_batch(customers: list[CustomerInput]):
    """Predict churn for multiple customers at once"""
    results = []
    for i, customer in enumerate(customers):
        try:
            df = preprocess_input(customer)
            churn_prob = float(model.predict_proba(df)[:, 1][0])
            churn_prediction = int(churn_prob >= 0.5)
            projected_clv = round(customer.MonthlyCharges * 12 * (1 - churn_prob), 2)
            
            if churn_prob >= 0.7:
                risk_level = "Critical"
            elif churn_prob >= 0.5:
                risk_level = "High"
            elif churn_prob >= 0.3:
                risk_level = "Medium"
            else:
                risk_level = "Low"

            strategy = get_strategy(churn_prob, customer.MonthlyCharges, customer.tenure)

            results.append({
                "customer_index": i + 1,
                "churn_probability": round(churn_prob * 100, 1),
                "churn_prediction": "Will Churn" if churn_prediction == 1 else "Will Stay",
                "risk_level": risk_level,
                "projected_12m_clv": projected_clv,
                "strategy": strategy
            })
        except Exception as e:
            results.append({"customer_index": i + 1, "error": str(e)})
    
    return {"total_customers": len(customers), "predictions": results}

@app.post("/debug")
def debug_input(customer: CustomerInput):
    df = preprocess_input(customer)
    return {
        "received_input": customer.dict(),
        "processed_features": df.iloc[0].to_dict(),
        "prediction": float(model.predict_proba(df)[:, 1][0]) * 100
    }