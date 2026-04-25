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
    
    input_dict = {
        'gender': data.gender,
        'SeniorCitizen': data.SeniorCitizen,
        'Partner': data.Partner,
        'Dependents': data.Dependents,
        'tenure': data.tenure,
        'PhoneService': data.PhoneService,
        'MultipleLines': data.MultipleLines,
        'InternetService': data.InternetService,
        'OnlineSecurity': data.OnlineSecurity,
        'OnlineBackup': data.OnlineBackup,
        'DeviceProtection': data.DeviceProtection,
        'TechSupport': data.TechSupport,
        'StreamingTV': data.StreamingTV,
        'StreamingMovies': data.StreamingMovies,
        'Contract': data.Contract,
        'PaperlessBilling': data.PaperlessBilling,
        'PaymentMethod': data.PaymentMethod,
        'MonthlyCharges': data.MonthlyCharges,
        'TotalCharges': data.TotalCharges
    }
    
    df = pd.DataFrame([input_dict])
    
    # Add engineered binary features
    df['MultipleLines_bin'] = df['MultipleLines'].map(
        {'Yes': 1, 'No': 0, 'No phone service': 0})
    df['InternetService_bin'] = df['InternetService'].map(
        {'Fiber optic': 1, 'DSL': 1, 'No': 0})
    df['OnlineSecurity_bin'] = df['OnlineSecurity'].map(
        {'Yes': 1, 'No': 0, 'No internet service': 0})
    df['OnlineBackup_bin'] = df['OnlineBackup'].map(
        {'Yes': 1, 'No': 0, 'No internet service': 0})
    df['DeviceProtection_bin'] = df['DeviceProtection'].map(
        {'Yes': 1, 'No': 0, 'No internet service': 0})
    df['TechSupport_bin'] = df['TechSupport'].map(
        {'Yes': 1, 'No': 0, 'No internet service': 0})
    df['StreamingTV_bin'] = df['StreamingTV'].map(
        {'Yes': 1, 'No': 0, 'No internet service': 0})
    df['StreamingMovies_bin'] = df['StreamingMovies'].map(
        {'Yes': 1, 'No': 0, 'No internet service': 0})

    service_bin_cols = ['PhoneService', 'MultipleLines_bin', 'InternetService_bin',
                        'OnlineSecurity_bin', 'OnlineBackup_bin', 'DeviceProtection_bin',
                        'TechSupport_bin', 'StreamingTV_bin', 'StreamingMovies_bin']
    df['num_services'] = df[service_bin_cols].sum(axis=1)

    # Encode categorical columns
    cat_cols = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Reorder columns to exactly match training order
    final_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                  'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                  'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                  'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                  'MonthlyCharges', 'TotalCharges', 'MultipleLines_bin',
                  'InternetService_bin', 'OnlineSecurity_bin', 'OnlineBackup_bin',
                  'DeviceProtection_bin', 'TechSupport_bin', 'StreamingTV_bin',
                  'StreamingMovies_bin', 'num_services']
    
    return df[final_cols]


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