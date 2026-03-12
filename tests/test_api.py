"""
Unit Tests — Travel MLOps Flask API
Run: pytest tests/ -v
"""

import pytest
import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flask_api.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# ─── Health & Root ────────────────────────────────────────────

def test_home(client):
    r = client.get('/')
    assert r.status_code == 200
    data = json.loads(r.data)
    assert 'endpoints' in data

def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data['status'] == 'healthy'

def test_metadata_regression(client):
    r = client.get('/metadata/regression')
    assert r.status_code == 200
    data = json.loads(r.data)
    assert 'features' in data
    assert 'metrics' in data

def test_metadata_classification(client):
    r = client.get('/metadata/classification')
    assert r.status_code == 200
    data = json.loads(r.data)
    assert 'features' in data

# ─── Regression Endpoint ──────────────────────────────────────

def test_predict_flight_price_valid(client):
    payload = {
        "from": "Recife (PE)",
        "to": "Florianopolis (SC)",
        "flightType": "firstClass",
        "time": 2.5,
        "distance": 700,
        "agency": "FlyingDrops",
        "month": 9,
        "dayofweek": 3
    }
    r = client.post('/predict/flight-price',
                    data=json.dumps(payload),
                    content_type='application/json')
    assert r.status_code == 200
    data = json.loads(r.data)
    assert 'predicted_price' in data
    assert data['predicted_price'] > 0

def test_predict_flight_price_missing_field(client):
    payload = {"from": "Recife (PE)", "to": "Florianopolis (SC)"}
    r = client.post('/predict/flight-price',
                    data=json.dumps(payload),
                    content_type='application/json')
    assert r.status_code == 400
    data = json.loads(r.data)
    assert 'error' in data

# ─── Classification Endpoint ──────────────────────────────────

def test_predict_gender_valid(client):
    payload = {
        "age": 30,
        "from": "Recife (PE)",
        "to": "Florianopolis (SC)",
        "flightType": "economic",
        "price": 500.0,
        "time": 2.5,
        "distance": 700,
        "agency": "FlyingDrops",
        "month": 6
    }
    r = client.post('/predict/gender',
                    data=json.dumps(payload),
                    content_type='application/json')
    assert r.status_code == 200
    data = json.loads(r.data)
    assert 'predicted_gender' in data
    assert data['predicted_gender'] in ['male', 'female']
    assert 'confidence_percent' in data

def test_predict_gender_missing_field(client):
    payload = {"age": 25}
    r = client.post('/predict/gender',
                    data=json.dumps(payload),
                    content_type='application/json')
    assert r.status_code == 400
