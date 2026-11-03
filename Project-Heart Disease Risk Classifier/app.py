# Import necessary libraries
import os
import math
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from load_data import CATEGORICAL_COLUMNS

# Load the trained model once, when the server starts (run train_final_model.py first).
# Loading needs select_columns from load_data.py, which is why the app lives in this folder.
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")
if not os.path.exists(MODEL_FILE):
    raise SystemExit("model.joblib not found. Run 'python train_final_model.py' first.")

bundle = joblib.load(MODEL_FILE)
model = bundle['model']
THRESHOLD = bundle['threshold']
RAW_FEATURES = bundle['raw_features']        # the inputs this model needs (decided at training time)
AVERAGE_FEATURES = bundle['average_features']

# What each of the 13 possible inputs may contain. The app only asks for (and checks)
# the ones in RAW_FEATURES, so a retrained model with different features just works.
CODED_INPUTS = {   # input -> {code: readable name}
    'sex': {0: 'Female', 1: 'Male'},
    'cp': {1: 'Typical angina', 2: 'Atypical angina', 3: 'Non-anginal pain', 4: 'Asymptomatic'},
    'fbs': {0: 'No', 1: 'Yes'},
    'restecg': {0: 'Normal', 1: 'ST-T abnormality', 2: 'LV hypertrophy'},
    'exang': {0: 'No', 1: 'Yes'},
    'slope': {1: 'Upsloping', 2: 'Flat', 3: 'Downsloping'},
    'ca': {0: '0', 1: '1', 2: '2', 3: '3'},
    'thal': {3: 'Normal', 6: 'Fixed defect', 7: 'Reversible defect'}
}
NUMERIC_INPUTS = {  # input -> (min, max), a little wider than the training data
    'age': (18, 100),
    'trestbps': (80, 220),
    'chol': (100, 600),
    'thalach': (60, 220),
    'oldpeak': (0.0, 6.5)
}

# Readable names for the model's features (shown in the explanation)
INPUT_NAMES = {
    'age': 'Age', 'sex': 'Sex', 'cp': 'Chest pain', 'trestbps': 'Resting blood pressure',
    'chol': 'Cholesterol', 'fbs': 'High fasting blood sugar', 'restecg': 'Resting ECG',
    'thalach': 'Max heart rate', 'exang': 'Exercise-induced angina', 'oldpeak': 'ST depression',
    'slope': 'ST slope', 'ca': 'Blocked major vessels', 'thal': 'Thallium scan'
}


def feature_label(feature):
    """'cp_4' -> 'Chest pain: Asymptomatic', 'sex' -> 'Male sex', 'ca' -> 'Blocked major vessels'."""
    prefix, _, code = feature.rpartition('_')
    if prefix in CATEGORICAL_COLUMNS:
        return f"{INPUT_NAMES[prefix]}: {CODED_INPUTS[prefix][int(code)]}"
    if feature == 'sex':
        return 'Male sex'
    return INPUT_NAMES[feature]


app = Flask(__name__)


def validate(data):
    """Check the request body against the inputs this model needs. Returns (patient, errors)."""
    if not isinstance(data, dict):
        return None, ["Send a JSON object with the patient's values."]

    patient, errors = {}, []
    for name in RAW_FEATURES:
        value = data.get(name)
        # bool is a subclass of int in Python, so reject true/false explicitly
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"'{name}' is missing or not a number.")
            continue

        if name in NUMERIC_INPUTS:
            low, high = NUMERIC_INPUTS[name]
            if not math.isfinite(value) or not low <= value <= high:
                errors.append(f"'{name}' must be between {low} and {high}.")
                continue
            patient[name] = float(value)
        else:
            allowed = list(CODED_INPUTS[name])
            if value not in allowed:
                errors.append(f"'{name}' must be one of {allowed}.")
                continue
            patient[name] = int(value)

    return patient, errors


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/model-info")
def model_info():
    """Facts about the current model, including which inputs the UI should ask for."""
    return jsonify({
        'raw_features': RAW_FEATURES,
        'features': bundle['features'],
        'threshold': THRESHOLD,
        'target_recall': bundle['target_recall'],
        'test_metrics': bundle['test_metrics'],
        'n_patients': bundle['n_patients'],
        'trained_at': bundle['trained_at']
    })


@app.post("/api/predict")
def predict():
    """Predict heart disease risk for one patient."""
    patient, errors = validate(request.get_json(silent=True))
    if errors:
        return jsonify({'errors': errors}), 400

    row = pd.DataFrame([patient])[RAW_FEATURES]
    probability = float(model.predict_proba(row)[0, 1])

    # Explain the prediction: how far each feature moves the log-odds compared with the
    # average patient. Positive = pushes the risk up, negative = pushes it down.
    features = model[:-1].transform(row).iloc[0]
    coefficients = model[-1].coef_[0]
    contributions = [
        {
            'feature': name,
            'label': feature_label(name),
            'contribution': round(float(coef * (features[name] - AVERAGE_FEATURES[name])), 4)
        }
        for name, coef in zip(features.index, coefficients)
    ]
    contributions.sort(key=lambda item: -abs(item['contribution']))

    return jsonify({
        'probability': round(probability, 4),
        'threshold': THRESHOLD,
        'risk': 'high' if probability >= THRESHOLD else 'low',
        'contributions': contributions
    })


if __name__ == "__main__":
    app.run(debug=True)
