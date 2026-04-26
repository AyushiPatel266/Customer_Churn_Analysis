from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

FASTAPI_URL = "https://customer-churn-api.onrender.com"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form = request.form
        
        customer_data = {
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

        response = requests.post(f"{FASTAPI_URL}/predict", json=customer_data)
        result = response.json()
        
        return render_template('index.html', result=result, form_data=form)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, port=5000)