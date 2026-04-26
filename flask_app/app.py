from flask import Flask, render_template, request, jsonify
import requests
import time

app = Flask(__name__)

FASTAPI_URL = "https://customer-churn-api-spcg.onrender.com"

def wake_up_api():
    """Wake up FastAPI and wait until it is ready"""
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{FASTAPI_URL}/health", timeout=30)
            if response.status_code == 200:
                return True
        except:
            time.sleep(5)
    return False

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Wake up API and wait for it
        api_ready = wake_up_api()
        if not api_ready:
            return render_template('index.html',
                error="Server is waking up. Please wait 30 seconds and try again.")

        form = request.form
        customer_data = {
            # your existing code stays the same
            "gender": form.get('gender'),
            "SeniorCitizen": int(form.get('SeniorCitizen')),
            "Partner": int(form.get('Partner')),
            "Dependents": int(form.get('Dependents')),
            "tenure": int(form.get('tenure')),
            "PhoneService": int(form.get('PhoneService')),
            "MultipleLines": form.get('MultipleLines'),
            "InternetService": form.get('InternetService'),
            "OnlineSecurity": form.get('OnlineSecurity'),
            "OnlineBackup": form.get('OnlineBackup'),
            "DeviceProtection": form.get('DeviceProtection'),
            "TechSupport": form.get('TechSupport'),
            "StreamingTV": form.get('StreamingTV'),
            "StreamingMovies": form.get('StreamingMovies'),
            "Contract": form.get('Contract'),
            "PaperlessBilling": int(form.get('PaperlessBilling')),
            "PaymentMethod": form.get('PaymentMethod'),
            "MonthlyCharges": float(form.get('MonthlyCharges')),
            "TotalCharges": float(form.get('TotalCharges'))
        }

        response = requests.post(
            f"{FASTAPI_URL}/predict",
            json=customer_data,
            timeout=120
        )
        result = response.json()
        return render_template('index.html', result=result, form_data=form)

    except Exception as e:
        return render_template('index.html', error=str(e))

