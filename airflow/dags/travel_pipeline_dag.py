"""
Apache Airflow DAG — Travel MLOps Pipeline
Orchestrates: Data ingestion → Preprocessing → Model Training → Evaluation → Deployment
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import os
import logging

# ─── Default Arguments ────────────────────────────────────────────────────────
default_args = {
    'owner': 'travel_mlops',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

PROJECT_PATH = os.environ.get('TRAVEL_PROJECT_PATH', '/opt/airflow/travel_mlops')

# ─── Task Functions ───────────────────────────────────────────────────────────

def data_ingestion(**context):
    """Load and validate raw CSV datasets."""
    import pandas as pd
    logging.info("Starting data ingestion...")
    
    flights = pd.read_csv(f'{PROJECT_PATH}/data/flights.csv')
    hotels  = pd.read_csv(f'{PROJECT_PATH}/data/hotels.csv')
    users   = pd.read_csv(f'{PROJECT_PATH}/data/users.csv')
    
    assert len(flights) > 0, "Flights data is empty!"
    assert len(hotels) > 0,  "Hotels data is empty!"
    assert len(users) > 0,   "Users data is empty!"
    
    logging.info(f"Loaded: {len(flights)} flights, {len(hotels)} hotels, {len(users)} users")
    
    # Push stats to XCom
    context['ti'].xcom_push(key='row_counts', value={
        'flights': len(flights),
        'hotels': len(hotels),
        'users': len(users)
    })
    return "Data ingestion completed"


def data_preprocessing(**context):
    """Clean and preprocess data for model training."""
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    import joblib
    
    logging.info("Starting data preprocessing...")
    
    flights = pd.read_csv(f'{PROJECT_PATH}/data/flights.csv')
    flights['date'] = pd.to_datetime(flights['date'], format='%m/%d/%Y')
    flights['month'] = flights['date'].dt.month
    flights['dayofweek'] = flights['date'].dt.dayofweek
    
    # Remove outliers using IQR
    Q1 = flights['price'].quantile(0.25)
    Q3 = flights['price'].quantile(0.75)
    IQR = Q3 - Q1
    initial_count = len(flights)
    flights = flights[
        (flights['price'] >= Q1 - 1.5*IQR) &
        (flights['price'] <= Q3 + 1.5*IQR)
    ]
    logging.info(f"Removed {initial_count - len(flights)} outliers")
    
    # Save preprocessed data
    flights.to_csv(f'{PROJECT_PATH}/data/flights_processed.csv', index=False)
    
    context['ti'].xcom_push(key='processed_rows', value=len(flights))
    logging.info(f"Preprocessing done. {len(flights)} records saved.")
    return "Preprocessing completed"


def train_regression_model(**context):
    """Train the flight price regression model."""
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import joblib, json
    
    logging.info("Training regression model...")
    
    flights = pd.read_csv(f'{PROJECT_PATH}/data/flights.csv')
    flights['date'] = pd.to_datetime(flights['date'], format='%m/%d/%Y')
    flights['month'] = flights['date'].dt.month
    flights['dayofweek'] = flights['date'].dt.dayofweek
    
    le_from   = LabelEncoder(); flights['from_enc']       = le_from.fit_transform(flights['from'])
    le_to     = LabelEncoder(); flights['to_enc']         = le_to.fit_transform(flights['to'])
    le_type   = LabelEncoder(); flights['flightType_enc'] = le_type.fit_transform(flights['flightType'])
    le_agency = LabelEncoder(); flights['agency_enc']     = le_agency.fit_transform(flights['agency'])
    
    features = ['from_enc','to_enc','flightType_enc','time','distance','agency_enc','month','dayofweek']
    X = flights[features]
    y = flights['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    
    # Save model and encoders
    joblib.dump(model,    f'{PROJECT_PATH}/models/flight_price_model.pkl')
    joblib.dump(le_from,  f'{PROJECT_PATH}/models/le_from.pkl')
    joblib.dump(le_to,    f'{PROJECT_PATH}/models/le_to.pkl')
    joblib.dump(le_type,  f'{PROJECT_PATH}/models/le_flighttype.pkl')
    joblib.dump(le_agency,f'{PROJECT_PATH}/models/le_agency.pkl')
    
    metrics = {'rmse': round(rmse, 4), 'r2': round(r2, 4), 'mae': round(mae, 4)}
    context['ti'].xcom_push(key='regression_metrics', value=metrics)
    
    logging.info(f"Regression model trained — RMSE={rmse:.4f}, R2={r2:.4f}")
    return f"Regression: RMSE={rmse:.4f}, R2={r2:.4f}"


def train_classification_model(**context):
    """Train the gender classification model."""
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib
    
    logging.info("Training classification model...")
    
    flights = pd.read_csv(f'{PROJECT_PATH}/data/flights.csv')
    users   = pd.read_csv(f'{PROJECT_PATH}/data/users.csv')
    merged  = flights.merge(users, left_on='userCode', right_on='code')
    merged['date'] = pd.to_datetime(merged['date'], format='%m/%d/%Y')
    merged['month'] = merged['date'].dt.month
    
    clf_df = merged[merged['gender'] != 'none'].copy()
    
    for col, le_name in [('from', 'clf_le_from'), ('to', 'clf_le_to'),
                          ('flightType', 'clf_le_flighttype'), ('agency', 'clf_le_agency')]:
        le = LabelEncoder()
        clf_df[f'{col}_enc'] = le.fit_transform(clf_df[col])
        joblib.dump(le, f'{PROJECT_PATH}/models/{le_name}.pkl')
    
    le_gender = LabelEncoder()
    clf_df['gender_enc'] = le_gender.fit_transform(clf_df['gender'])
    
    features = ['age','from_enc','to_enc','flightType_enc','price','time','distance','agency_enc','month']
    X = clf_df[features]
    y = clf_df['gender_enc']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test_s))
    
    joblib.dump(clf,      f'{PROJECT_PATH}/models/gender_classifier.pkl')
    joblib.dump(scaler,   f'{PROJECT_PATH}/models/gender_scaler.pkl')
    joblib.dump(le_gender,f'{PROJECT_PATH}/models/le_gender.pkl')
    
    context['ti'].xcom_push(key='classification_accuracy', value=round(acc, 4))
    logging.info(f"Classification model trained — Accuracy={acc:.4f}")
    return f"Classification Accuracy={acc:.4f}"


def evaluate_and_report(**context):
    """Evaluate models and generate performance report."""
    ti = context['ti']
    
    reg_metrics = ti.xcom_pull(task_ids='train_regression', key='regression_metrics')
    clf_acc     = ti.xcom_pull(task_ids='train_classification', key='classification_accuracy')
    row_counts  = ti.xcom_pull(task_ids='data_ingestion', key='row_counts')
    
    report = f"""
    ================================
    TRAVEL MLOPS — PIPELINE REPORT
    ================================
    Run Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    DATA STATS:
      Flights : {row_counts.get('flights', 'N/A'):,} records
      Hotels  : {row_counts.get('hotels', 'N/A'):,} records
      Users   : {row_counts.get('users', 'N/A'):,} records
    
    REGRESSION MODEL (Flight Price):
      RMSE    : {reg_metrics.get('rmse', 'N/A')}
      R²      : {reg_metrics.get('r2', 'N/A')}
      MAE     : {reg_metrics.get('mae', 'N/A')}
    
    CLASSIFICATION MODEL (Gender):
      Accuracy: {clf_acc}
    ================================
    """
    
    logging.info(report)
    
    with open(f'{PROJECT_PATH}/docs/pipeline_report.txt', 'w') as f:
        f.write(report)
    
    return "Report generated"


# ─── DAG Definition ───────────────────────────────────────────────────────────
with DAG(
    dag_id='travel_mlops_pipeline',
    default_args=default_args,
    description='End-to-end Travel MLOps Pipeline: ingest → preprocess → train → evaluate',
    schedule_interval='@daily',
    catchup=False,
    tags=['travel', 'mlops', 'regression', 'classification'],
) as dag:

    t1 = PythonOperator(
        task_id='data_ingestion',
        python_callable=data_ingestion,
        provide_context=True,
    )

    t2 = PythonOperator(
        task_id='data_preprocessing',
        python_callable=data_preprocessing,
        provide_context=True,
    )

    t3 = PythonOperator(
        task_id='train_regression',
        python_callable=train_regression_model,
        provide_context=True,
    )

    t4 = PythonOperator(
        task_id='train_classification',
        python_callable=train_classification_model,
        provide_context=True,
    )

    t5 = PythonOperator(
        task_id='evaluate_and_report',
        python_callable=evaluate_and_report,
        provide_context=True,
    )

    t6 = BashOperator(
        task_id='restart_api_service',
        bash_command='echo "API restart triggered — run: docker restart travel_mlops_api"',
    )

    # Pipeline dependency chain
    t1 >> t2 >> [t3, t4] >> t5 >> t6
