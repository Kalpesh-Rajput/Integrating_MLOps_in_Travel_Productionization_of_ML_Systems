"""
MLflow Model Tracking — Travel MLOps
Tracks experiments, parameters, metrics, and model artifacts for both models.
Run: python mlflow_tracking/mlflow_train.py
Then: mlflow ui  (opens at http://localhost:5000)
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                              accuracy_score, precision_score, recall_score, f1_score)
import joblib, os, json

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE, '..', 'data')
MODELS_PATH = os.path.join(BASE, '..', 'models')

# ─── Set MLflow Tracking URI ─────────────────────────────────────────────────
mlflow.set_tracking_uri("sqlite:///mlflow_tracking/mlruns.db")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: REGRESSION — FLIGHT PRICE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_regression_experiments():
    mlflow.set_experiment("Flight_Price_Prediction")
    
    flights = pd.read_csv(os.path.join(DATA_PATH, 'flights.csv'))
    flights['date'] = pd.to_datetime(flights['date'], format='%m/%d/%Y')
    flights['month']      = flights['date'].dt.month
    flights['dayofweek']  = flights['date'].dt.dayofweek

    le_from   = LabelEncoder(); flights['from_enc']       = le_from.fit_transform(flights['from'])
    le_to     = LabelEncoder(); flights['to_enc']         = le_to.fit_transform(flights['to'])
    le_type   = LabelEncoder(); flights['flightType_enc'] = le_type.fit_transform(flights['flightType'])
    le_agency = LabelEncoder(); flights['agency_enc']     = le_agency.fit_transform(flights['agency'])

    features = ['from_enc','to_enc','flightType_enc','time','distance','agency_enc','month','dayofweek']
    X = flights[features]
    y = flights['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ── Run 1: Linear Regression ─────────────────────────────────────────────
    with mlflow.start_run(run_name="LinearRegression_Baseline"):
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("features", str(features))
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(model, "linear_regression_model")
        
        print(f"[Linear Regression]  RMSE={rmse:.4f}  R2={r2:.4f}  MAE={mae:.4f}")

    # ── Run 2: Random Forest (n=50) ───────────────────────────────────────────
    with mlflow.start_run(run_name="RandomForest_n50"):
        params = {'n_estimators': 50, 'max_depth': 10, 'random_state': 42}
        model = RandomForestRegressor(**params, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        
        mlflow.log_params(params)
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(model, "rf_regressor_n50")
        
        print(f"[RandomForest n=50]  RMSE={rmse:.4f}  R2={r2:.4f}  MAE={mae:.4f}")

    # ── Run 3: Random Forest (n=100) — Best Model ─────────────────────────────
    with mlflow.start_run(run_name="RandomForest_n100_BEST"):
        params = {'n_estimators': 100, 'max_depth': None, 'random_state': 42}
        model = RandomForestRegressor(**params, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        
        mlflow.log_params(params)
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mae", mae)
        
        # Log feature importances
        for feat, imp in zip(features, model.feature_importances_):
            mlflow.log_metric(f"importance_{feat}", round(float(imp), 4))
        
        mlflow.sklearn.log_model(model, "rf_regressor_n100",
                                  registered_model_name="FlightPricePredictor")
        
        print(f"[RandomForest n=100] RMSE={rmse:.4f}  R2={r2:.4f}  MAE={mae:.4f}  *** BEST ***")

    # ── Run 4: Gradient Boosting ──────────────────────────────────────────────
    with mlflow.start_run(run_name="GradientBoosting"):
        params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'random_state': 42}
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        
        mlflow.log_params(params)
        mlflow.log_param("model_type", "GradientBoostingRegressor")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(model, "gb_regressor")
        
        print(f"[GradientBoosting]   RMSE={rmse:.4f}  R2={r2:.4f}  MAE={mae:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: CLASSIFICATION — GENDER PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_classification_experiments():
    mlflow.set_experiment("Gender_Classification")
    
    flights = pd.read_csv(os.path.join(DATA_PATH, 'flights.csv'))
    users   = pd.read_csv(os.path.join(DATA_PATH, 'users.csv'))
    merged  = flights.merge(users, left_on='userCode', right_on='code')
    merged['date'] = pd.to_datetime(merged['date'], format='%m/%d/%Y')
    merged['month'] = merged['date'].dt.month
    
    clf_df = merged[merged['gender'] != 'none'].copy()
    
    le_from   = LabelEncoder(); clf_df['from_enc']       = le_from.fit_transform(clf_df['from'])
    le_to     = LabelEncoder(); clf_df['to_enc']         = le_to.fit_transform(clf_df['to'])
    le_type   = LabelEncoder(); clf_df['flightType_enc'] = le_type.fit_transform(clf_df['flightType'])
    le_agency = LabelEncoder(); clf_df['agency_enc']     = le_agency.fit_transform(clf_df['agency'])
    le_gender = LabelEncoder(); clf_df['gender_enc']     = le_gender.fit_transform(clf_df['gender'])
    
    features = ['age','from_enc','to_enc','flightType_enc','price','time','distance','agency_enc','month']
    X = clf_df[features]
    y = clf_df['gender_enc']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    
    def log_clf_metrics(y_true, y_pred, prefix=""):
        mlflow.log_metric("accuracy",  accuracy_score(y_true, y_pred))
        mlflow.log_metric("precision", precision_score(y_true, y_pred, average='macro'))
        mlflow.log_metric("recall",    recall_score(y_true, y_pred, average='macro'))
        mlflow.log_metric("f1_score",  f1_score(y_true, y_pred, average='macro'))
    
    # ── Run 1: Logistic Regression ────────────────────────────────────────────
    with mlflow.start_run(run_name="LogisticRegression_Baseline"):
        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 500)
        log_clf_metrics(y_test, y_pred)
        mlflow.sklearn.log_model(model, "logistic_regression")
        print(f"[LogisticRegression] Accuracy={accuracy_score(y_test, y_pred):.4f}")
    
    # ── Run 2: Random Forest Classifier ──────────────────────────────────────
    with mlflow.start_run(run_name="RandomForest_Classifier_BEST"):
        params = {'n_estimators': 100, 'random_state': 42}
        model = RandomForestClassifier(**params, n_jobs=-1)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_params(params)
        mlflow.log_param("model_type", "RandomForestClassifier")
        log_clf_metrics(y_test, y_pred)
        mlflow.sklearn.log_model(model, "rf_classifier",
                                  registered_model_name="GenderClassifier")
        print(f"[RandomForest Clf]   Accuracy={acc:.4f}  *** BEST ***")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  TRAVEL MLOPS — MLflow Experiment Tracking")
    print("=" * 60)
    
    print("\n[1/2] Running Regression Experiments...")
    run_regression_experiments()
    
    print("\n[2/2] Running Classification Experiments...")
    run_classification_experiments()
    
    print("\n✅ All experiments logged!")
    print("👉 Run 'mlflow ui' to view results at http://localhost:5000")
