───────────────────────────────────────────────────────────────────────────# ============================================================
# CELL 1: Mount Drive & Install Dependencies
# ============================================================
"""
# Run this cell FIRST in Google Colab
from google.colab import drive
drive.mount('/content/drive')

!pip install mlflow scikit-learn pandas numpy matplotlib seaborn -q
"""

# ============================================================
# CELL 2: Import Libraries
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                              accuracy_score, classification_report, confusion_matrix)
import joblib
import os

print("All libraries imported successfully!")

# ============================================================
# CELL 3: Load Datasets
# ============================================================
# If using Colab: upload files or read from Drive
# flights = pd.read_csv('/content/drive/MyDrive/travel_capstone/flights.csv')
# hotels  = pd.read_csv('/content/drive/MyDrive/travel_capstone/hotels.csv')
# users   = pd.read_csv('/content/drive/MyDrive/travel_capstone/users.csv')

# For local use:
flights = pd.read_csv('data/flights.csv')
hotels  = pd.read_csv('data/hotels.csv')
users   = pd.read_csv('data/users.csv')

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"\nFlights  : {flights.shape[0]:,} rows × {flights.shape[1]} columns")
print(f"Hotels   : {hotels.shape[0]:,} rows × {hotels.shape[1]} columns")
print(f"Users    : {users.shape[0]:,} rows × {users.shape[1]} columns")

# ============================================================
# CELL 4: Exploratory Data Analysis (EDA)
# ============================================================
print("\n" + "=" * 60)
print("EDA — FLIGHTS DATASET")
print("=" * 60)
print(flights.head())
print("\nData Types:\n", flights.dtypes)
print("\nMissing Values:\n", flights.isnull().sum())
print("\nFlight Types:", flights['flightType'].unique())
print("Agencies   :", flights['agency'].unique())
print("\nPrice Statistics:")
print(flights['price'].describe())

print("\n" + "=" * 60)
print("EDA — USERS DATASET")
print("=" * 60)
print(users.head())
print("\nGender Distribution:\n", users['gender'].value_counts())
print("\nAge Statistics:")
print(users['age'].describe())

print("\n" + "=" * 60)
print("EDA — HOTELS DATASET")
print("=" * 60)
print(hotels.head())
print("\nTop Hotel Destinations:")
print(hotels['place'].value_counts().head(10))

# ============================================================
# CELL 5: EDA Visualizations
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Travel MLOps — Exploratory Data Analysis', fontsize=16, fontweight='bold')

# 1. Flight type distribution
flights['flightType'].value_counts().plot(kind='pie', ax=axes[0,0], autopct='%1.1f%%',
    colors=['#4CAF50','#2196F3','#FF9800'])
axes[0,0].set_title('Flight Type Distribution')
axes[0,0].set_ylabel('')

# 2. Price distribution
axes[0,1].hist(flights['price'], bins=50, color='#2196F3', edgecolor='white', alpha=0.8)
axes[0,1].set_title('Flight Price Distribution')
axes[0,1].set_xlabel('Price (USD)')
axes[0,1].set_ylabel('Frequency')

# 3. Distance vs Price
sample = flights.sample(min(3000, len(flights)), random_state=42)
colors_map = {'firstClass': '#4CAF50', 'premium': '#FF9800', 'economic': '#2196F3'}
for ftype, grp in sample.groupby('flightType'):
    axes[0,2].scatter(grp['distance'], grp['price'], alpha=0.3, s=5,
                      color=colors_map.get(ftype,'gray'), label=ftype)
axes[0,2].set_title('Distance vs Price (by Flight Type)')
axes[0,2].set_xlabel('Distance (km)')
axes[0,2].set_ylabel('Price (USD)')
axes[0,2].legend()

# 4. Gender distribution
users['gender'].value_counts().plot(kind='bar', ax=axes[1,0],
    color=['#E91E63','#2196F3','#9E9E9E'])
axes[1,0].set_title('User Gender Distribution')
axes[1,0].set_xlabel('Gender')
axes[1,0].set_ylabel('Count')
axes[1,0].tick_params(axis='x', rotation=0)

# 5. Agency price comparison
flights.groupby('agency')['price'].mean().plot(kind='bar', ax=axes[1,1],
    color=['#673AB7','#E91E63','#00BCD4'])
axes[1,1].set_title('Average Price by Agency')
axes[1,1].set_xlabel('Agency')
axes[1,1].set_ylabel('Avg Price (USD)')
axes[1,1].tick_params(axis='x', rotation=0)

# 6. Hotel stay duration
hotels['days'].value_counts().sort_index().plot(kind='bar', ax=axes[1,2],
    color='#FF5722', alpha=0.8)
axes[1,2].set_title('Hotel Stay Duration Distribution')
axes[1,2].set_xlabel('Days')
axes[1,2].set_ylabel('Count')
axes[1,2].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('docs/eda_visualizations.png', dpi=150, bbox_inches='tight')
plt.show()
print("EDA visualizations saved.")

# ============================================================
# CELL 6: Feature Engineering
# ============================================================
print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# ── Regression Features ──────────────────────────────────────
df_reg = flights.copy()
df_reg['date'] = pd.to_datetime(df_reg['date'], format='%m/%d/%Y')
df_reg['month']      = df_reg['date'].dt.month
df_reg['dayofweek']  = df_reg['date'].dt.dayofweek
df_reg['year']       = df_reg['date'].dt.year

le_from   = LabelEncoder()
le_to     = LabelEncoder()
le_type   = LabelEncoder()
le_agency = LabelEncoder()

df_reg['from_enc']       = le_from.fit_transform(df_reg['from'])
df_reg['to_enc']         = le_to.fit_transform(df_reg['to'])
df_reg['flightType_enc'] = le_type.fit_transform(df_reg['flightType'])
df_reg['agency_enc']     = le_agency.fit_transform(df_reg['agency'])

REG_FEATURES = ['from_enc','to_enc','flightType_enc','time','distance','agency_enc','month','dayofweek']

print(f"Regression Features  : {REG_FEATURES}")
print(f"Target Variable      : price")
print(f"Total Training Rows  : {len(df_reg):,}")

# ── Classification Features ──────────────────────────────────
df_clf = flights.merge(users, left_on='userCode', right_on='code')
df_clf['date'] = pd.to_datetime(df_clf['date'], format='%m/%d/%Y')
df_clf['month'] = df_clf['date'].dt.month
df_clf = df_clf[df_clf['gender'] != 'none'].copy()

clf_le_from   = LabelEncoder()
clf_le_to     = LabelEncoder()
clf_le_type   = LabelEncoder()
clf_le_agency = LabelEncoder()
le_gender     = LabelEncoder()

df_clf['from_enc']       = clf_le_from.fit_transform(df_clf['from'])
df_clf['to_enc']         = clf_le_to.fit_transform(df_clf['to'])
df_clf['flightType_enc'] = clf_le_type.fit_transform(df_clf['flightType'])
df_clf['agency_enc']     = clf_le_agency.fit_transform(df_clf['agency'])
df_clf['gender_enc']     = le_gender.fit_transform(df_clf['gender'])

CLF_FEATURES = ['age','from_enc','to_enc','flightType_enc','price','time','distance','agency_enc','month']

print(f"\nClassification Features: {CLF_FEATURES}")
print(f"Target Variable       : gender ({le_gender.classes_})")
print(f"Total Training Rows   : {len(df_clf):,}")

# ============================================================
# CELL 7: REGRESSION MODEL — Flight Price Prediction
# ============================================================
print("\n" + "=" * 60)
print("REGRESSION MODEL — FLIGHT PRICE PREDICTION")
print("=" * 60)

X_reg = df_reg[REG_FEATURES]
y_reg = df_reg['price']

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

print(f"Training samples : {len(X_train):,}")
print(f"Test samples     : {len(X_test):,}")

# ── Model 1: Linear Regression (Baseline) ────────────────────
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr   = r2_score(y_test, y_pred_lr)
print(f"\nLinear Regression  — RMSE: {rmse_lr:.4f}  R²: {r2_lr:.4f}")

# ── Model 2: Random Forest Regressor (Best) ──────────────────
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf   = r2_score(y_test, y_pred_rf)
mae_rf  = mean_absolute_error(y_test, y_pred_rf)
print(f"Random Forest      — RMSE: {rmse_rf:.4f}  R²: {r2_rf:.4f}  MAE: {mae_rf:.4f}  *** BEST ***")

# ── Cross Validation ──────────────────────────────────────────
cv_scores = cross_val_score(rf_reg, X_reg, y_reg, cv=5, scoring='r2', n_jobs=-1)
print(f"\nCross-Validation R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Feature Importance Plot ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

feat_imp = pd.Series(rf_reg.feature_importances_, index=REG_FEATURES).sort_values(ascending=True)
feat_imp.plot(kind='barh', ax=axes[0], color='#2196F3', alpha=0.8)
axes[0].set_title('Feature Importances — Regression Model')
axes[0].set_xlabel('Importance Score')

# Actual vs Predicted
sample_idx = np.random.choice(len(y_test), 500)
axes[1].scatter(y_test.iloc[sample_idx], y_pred_rf[sample_idx], alpha=0.3, s=5, color='#4CAF50')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Price')
axes[1].set_ylabel('Predicted Price')
axes[1].set_title('Actual vs Predicted (Random Forest)')

plt.tight_layout()
plt.savefig('docs/regression_results.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n📊 Regression Summary:")
print(f"   Best Model : Random Forest Regressor")
print(f"   R² Score   : {r2_rf:.4f}")
print(f"   RMSE       : ${rmse_rf:.2f}")
print(f"   MAE        : ${mae_rf:.2f}")

# Save best model
joblib.dump(rf_reg,    'models/flight_price_model.pkl')
joblib.dump(le_from,   'models/le_from.pkl')
joblib.dump(le_to,     'models/le_to.pkl')
joblib.dump(le_type,   'models/le_flighttype.pkl')
joblib.dump(le_agency, 'models/le_agency.pkl')
print("\n✅ Regression model saved!")

# ============================================================
# CELL 8: CLASSIFICATION MODEL — Gender Prediction
# ============================================================
print("\n" + "=" * 60)
print("CLASSIFICATION MODEL — GENDER PREDICTION")
print("=" * 60)

X_clf = df_clf[CLF_FEATURES]
y_clf = df_clf['gender_enc']

X_tr, X_te, y_tr, y_te = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)

print(f"Training samples : {len(X_tr):,}")
print(f"Test samples     : {len(X_te):,}")
print(f"Class distribution: {dict(pd.Series(y_tr).value_counts())}")

# ── Model 1: Logistic Regression (Baseline) ──────────────────
log_reg = LogisticRegression(max_iter=500, random_state=42)
log_reg.fit(X_tr_s, y_tr)
acc_lr = accuracy_score(y_te, log_reg.predict(X_te_s))
print(f"\nLogistic Regression — Accuracy: {acc_lr:.4f}")

# ── Model 2: Random Forest Classifier (Best) ─────────────────
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_clf.fit(X_tr_s, y_tr)
y_pred_clf = rf_clf.predict(X_te_s)
acc_rf  = accuracy_score(y_te, y_pred_clf)
print(f"Random Forest Clf   — Accuracy: {acc_rf:.4f}  *** BEST ***")

print("\nDetailed Classification Report:")
print(classification_report(y_te, y_pred_clf, target_names=le_gender.classes_))

# ── Confusion Matrix ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm = confusion_matrix(y_te, y_pred_clf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=le_gender.classes_, yticklabels=le_gender.classes_)
axes[0].set_title('Confusion Matrix — Gender Classification')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Feature importance for classifier
feat_imp_clf = pd.Series(rf_clf.feature_importances_, index=CLF_FEATURES).sort_values(ascending=True)
feat_imp_clf.plot(kind='barh', ax=axes[1], color='#E91E63', alpha=0.8)
axes[1].set_title('Feature Importances — Classification Model')
axes[1].set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('docs/classification_results.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n📊 Classification Summary:")
print(f"   Best Model : Random Forest Classifier")
print(f"   Accuracy   : {acc_rf:.4f} ({acc_rf*100:.1f}%)")

# Save classification model
joblib.dump(rf_clf,       'models/gender_classifier.pkl')
joblib.dump(scaler,       'models/gender_scaler.pkl')
joblib.dump(le_gender,    'models/le_gender.pkl')
joblib.dump(clf_le_from,  'models/clf_le_from.pkl')
joblib.dump(clf_le_to,    'models/clf_le_to.pkl')
joblib.dump(clf_le_type,  'models/clf_le_flighttype.pkl')
joblib.dump(clf_le_agency,'models/clf_le_agency.pkl')
print("✅ Classification model saved!")

# ============================================================
# CELL 9: RECOMMENDATION MODEL — Hotel Recommendations
# ============================================================
print("\n" + "=" * 60)
print("RECOMMENDATION MODEL — HOTEL SUGGESTIONS")
print("=" * 60)

"""
Recommendation Approach:
  - Content-Based Filtering using hotel features
  - Score = weighted combination of booking count, price value, avg stay duration
  - Filtered by destination + budget constraints
"""

hotel_df = hotels.merge(users, left_on='userCode', right_on='code')
print(f"Merged hotel+user data: {hotel_df.shape}")

def recommend_hotels(destination, max_price_per_night, num_recommendations=5):
    """
    Recommend hotels based on destination and budget.
    
    Parameters:
        destination       : str  — city/place name
        max_price_per_night: float — maximum price per night in USD
        num_recommendations: int  — number of hotels to return
    
    Returns:
        DataFrame with recommended hotels and scores
    """
    filtered = hotels[
        (hotels['place'] == destination) &
        (hotels['price'] <= max_price_per_night)
    ].copy()
    
    if filtered.empty:
        return pd.DataFrame(columns=['name','avg_price','bookings','avg_stay','score'])
    
    hotel_stats = filtered.groupby('name').agg(
        avg_price=('price', 'mean'),
        bookings=('userCode', 'count'),
        avg_stay=('days', 'mean'),
        total_revenue=('total', 'sum')
    ).reset_index()
    
    # Normalize scores
    hotel_stats['price_score']    = 1 - (hotel_stats['avg_price'] / hotel_stats['avg_price'].max())
    hotel_stats['booking_score']  = hotel_stats['bookings'] / hotel_stats['bookings'].max()
    hotel_stats['stay_score']     = hotel_stats['avg_stay'] / hotel_stats['avg_stay'].max()
    
    # Weighted composite score
    hotel_stats['score'] = (
        hotel_stats['booking_score'] * 0.5 +
        hotel_stats['price_score']   * 0.3 +
        hotel_stats['stay_score']    * 0.2
    ).round(4)
    
    return hotel_stats.sort_values('score', ascending=False).head(num_recommendations)

# Demo recommendations
print("\n🏨 Top Hotels in Florianopolis (SC) — Budget: $500/night")
recs = recommend_hotels('Florianopolis (SC)', max_price_per_night=500)
print(recs[['name','avg_price','bookings','avg_stay','score']].to_string(index=False))

print("\n🏨 Top Hotels in Salvador (BH) — Budget: $400/night")
recs2 = recommend_hotels('Salvador (BH)', max_price_per_night=400)
print(recs2[['name','avg_price','bookings','avg_stay','score']].to_string(index=False))

# Visualize recommendations
fig, ax = plt.subplots(figsize=(10, 5))
if len(recs) > 0:
    ax.barh(recs['name'], recs['score'], color='#673AB7', alpha=0.85)
    ax.set_xlabel('Recommendation Score')
    ax.set_title('Hotel Recommendations — Florianopolis (SC)')
    for i, (score, price) in enumerate(zip(recs['score'], recs['avg_price'])):
        ax.text(score + 0.005, i, f'${price:.0f}/night', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig('docs/recommendation_results.png', dpi=150, bbox_inches='tight')
    plt.show()

print("\n✅ Recommendation model ready!")

# ============================================================
# CELL 10: Model Comparison Summary
# ============================================================
print("\n" + "=" * 60)
print("FINAL MODEL PERFORMANCE SUMMARY")
print("=" * 60)

summary = pd.DataFrame({
    'Model Type'   : ['Flight Price Regression', 'Gender Classification', 'Hotel Recommendation'],
    'Algorithm'    : ['Random Forest Regressor', 'Random Forest Classifier', 'Content-Based Filtering'],
    'Key Metric'   : ['R² Score', 'Accuracy', 'Precision@5'],
    'Score'        : [f'{r2_rf:.4f}', f'{acc_rf:.4f}', 'N/A (rank-based)'],
    'Status'       : ['✅ Saved', '✅ Saved', '✅ Function Ready'],
})
print(summary.to_string(index=False))

print("""
═══════════════════════════════════════════════════════════
PROJECT SUMMARY
═══════════════════════════════════════════════════════════
This notebook covers:
  1. Exploratory Data Analysis across 3 datasets
  2. Feature Engineering with label encoding + date features
  3. Regression Model (Random Forest) for flight price prediction
  4. Classification Model (Random Forest) for gender prediction
  5. Recommendation System for hotel suggestions

All models are saved as .pkl files for serving via Flask API.
Next steps: Docker → Kubernetes → Airflow → Jenkins → MLflow

GitHub Repo : https://github.com/[your-username]/travel-mlops-capstone
═══════════════════════════════════════════════════════════
""")
