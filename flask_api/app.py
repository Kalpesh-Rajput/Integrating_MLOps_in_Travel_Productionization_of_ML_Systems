"""
Travel MLOps - Flask REST API
Serves: Flight Price Prediction (Regression) + Gender Classification
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import json
import os

app = Flask(__name__)

# ─── Load Models & Encoders ─────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, '..', 'models')

reg_model    = joblib.load(os.path.join(MODELS, 'flight_price_model.pkl'))
le_from      = joblib.load(os.path.join(MODELS, 'le_from.pkl'))
le_to        = joblib.load(os.path.join(MODELS, 'le_to.pkl'))
le_type      = joblib.load(os.path.join(MODELS, 'le_flighttype.pkl'))
le_agency    = joblib.load(os.path.join(MODELS, 'le_agency.pkl'))

clf_model    = joblib.load(os.path.join(MODELS, 'gender_classifier.pkl'))
clf_scaler   = joblib.load(os.path.join(MODELS, 'gender_scaler.pkl'))
le_gender    = joblib.load(os.path.join(MODELS, 'le_gender.pkl'))
clf_le_from  = joblib.load(os.path.join(MODELS, 'clf_le_from.pkl'))
clf_le_to    = joblib.load(os.path.join(MODELS, 'clf_le_to.pkl'))
clf_le_type  = joblib.load(os.path.join(MODELS, 'clf_le_flighttype.pkl'))
clf_le_agency= joblib.load(os.path.join(MODELS, 'clf_le_agency.pkl'))

with open(os.path.join(MODELS, 'regression_meta.json')) as f:
    reg_meta = json.load(f)
with open(os.path.join(MODELS, 'classification_meta.json')) as f:
    clf_meta = json.load(f)

# ─── Helper: safe label encoding ────────────────────────────────────────────
def safe_encode(encoder, value):
    classes = list(encoder.classes_)
    if value in classes:
        return encoder.transform([value])[0]
    return 0  # default to first class if unseen

# ─── Routes ─────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Travel MLOps API is running!",
        "endpoints": {
            "GET  /health": "Health check",
            "POST /predict/flight-price": "Predict flight price",
            "POST /predict/gender": "Predict user gender",
            "GET  /metadata/regression": "Regression model metadata",
            "GET  /metadata/classification": "Classification model metadata"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "models_loaded": True}), 200

@app.route('/metadata/regression', methods=['GET'])
def reg_metadata():
    return jsonify(reg_meta)

@app.route('/metadata/classification', methods=['GET'])
def clf_metadata():
    return jsonify(clf_meta)

@app.route('/predict/flight-price', methods=['POST'])
def predict_flight_price():
    """
    Predicts flight price.
    Required JSON fields:
      from, to, flightType, time (hours), distance (km), agency, month (1-12), dayofweek (0-6)
    """
    try:
        data = request.get_json(force=True)
        required = ['from', 'to', 'flightType', 'time', 'distance', 'agency', 'month', 'dayofweek']
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        features = np.array([[
            safe_encode(le_from,   data['from']),
            safe_encode(le_to,     data['to']),
            safe_encode(le_type,   data['flightType']),
            float(data['time']),
            float(data['distance']),
            safe_encode(le_agency, data['agency']),
            int(data['month']),
            int(data['dayofweek'])
        ]])

        price = reg_model.predict(features)[0]

        return jsonify({
            "predicted_price": round(float(price), 2),
            "currency": "USD",
            "input": data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict/gender', methods=['POST'])
def predict_gender():
    """
    Predicts user gender based on flight & user features.
    Required JSON fields:
      age, from, to, flightType, price, time, distance, agency, month
    """
    try:
        data = request.get_json(force=True)
        required = ['age', 'from', 'to', 'flightType', 'price', 'time', 'distance', 'agency', 'month']
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        features = np.array([[
            float(data['age']),
            safe_encode(clf_le_from,   data['from']),
            safe_encode(clf_le_to,     data['to']),
            safe_encode(clf_le_type,   data['flightType']),
            float(data['price']),
            float(data['time']),
            float(data['distance']),
            safe_encode(clf_le_agency, data['agency']),
            int(data['month'])
        ]])

        features_scaled = clf_scaler.transform(features)
        pred_enc = clf_model.predict(features_scaled)[0]
        pred_proba = clf_model.predict_proba(features_scaled)[0]
        gender = le_gender.inverse_transform([pred_enc])[0]
        confidence = round(float(max(pred_proba)) * 100, 2)

        return jsonify({
            "predicted_gender": gender,
            "confidence_percent": confidence,
            "probabilities": {
                cls: round(float(prob)*100, 2)
                for cls, prob in zip(le_gender.classes_, pred_proba)
            },
            "input": data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
