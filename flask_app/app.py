from flask import Flask, render_template, request
import requests
import time
import os

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

FASTAPI_URL = "https://customer-churn-api-spcg.onrender.com"


def wake_up_api():
    """Wake up FastAPI and wait until it is ready"""
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{FASTAPI_URL}/health", timeout=60)
            if response.status_code == 200:
                return True
        except:
            time.sleep(10)
    return False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        api_ready = wake_up_api()
        if not api_ready:
            return render_template('index.html',
                error="Server is waking up. Please wait 30 seconds and try again.")

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

        response = requests.post(
            f"{FASTAPI_URL}/predict",
            json=customer_data,
            timeout=120
        )
        result = response.json()
        return render_template('index.html', result=result, form_data=form)

    except Exception as e:
        return render_template('index.html', error=str(e))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)