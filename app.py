from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('fetalai_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Handle both form data (from browser) and JSON (from API requests)
    if request.is_json:
        data = request.get_json(force=True)
    else:
        data = request.form.to_dict()
        # Convert form data values to float
        data = {key: float(value) for key, value in data.items()}

    features = np.array([list(data.values())])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    # Adjust prediction back to 1,2,3 for user
    prediction = prediction + 1
    # Map prediction to a label
    labels = {1: "Normal", 2: "Suspect", 3: "Pathological"}
    result = labels[prediction]
    message = "Normal fetal health." if prediction == 1 else "Potential concern detected. Consult a healthcare provider."

    if request.is_json:
        return jsonify({'fetal_health': int(prediction)})
    else:
        return render_template('result.html', result=result, prediction=prediction, message=message)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)