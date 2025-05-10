from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load model
model = joblib.load("xgboost_fraud_model.pkl")

# Expected features from training
FEATURES = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8',
            'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16',
            'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24',
            'V25', 'V26', 'V27', 'V28', 'Amount']

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        df = pd.DataFrame([input_data], columns=FEATURES)

        prediction = model.predict(df)[0]
        fraud_prob = model.predict_proba(df)[0][1]

        return jsonify({
            'prediction': int(prediction),
            'fraud_probability': round(float(fraud_prob), 5)
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
