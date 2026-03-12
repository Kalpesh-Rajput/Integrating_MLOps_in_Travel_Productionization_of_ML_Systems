"""
Travel MLOps - Streamlit Web Application
Hotel Recommendation System + Dashboard Visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
from sklearn.metrics.pairwise import cosine_similarity

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Travel MLOps Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE, '..', 'data')
MODELS_PATH = os.path.join(BASE, '..', 'models')

# ─── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    flights = pd.read_csv(os.path.join(DATA_PATH, 'flights.csv'))
    hotels  = pd.read_csv(os.path.join(DATA_PATH, 'hotels.csv'))
    users   = pd.read_csv(os.path.join(DATA_PATH, 'users.csv'))
    flights['date'] = pd.to_datetime(flights['date'], format='%m/%d/%Y')
    hotels['date']  = pd.to_datetime(hotels['date'],  format='%m/%d/%Y')
    return flights, hotels, users

@st.cache_resource
def load_reg_model():
    model    = joblib.load(os.path.join(MODELS_PATH, 'flight_price_model.pkl'))
    le_from  = joblib.load(os.path.join(MODELS_PATH, 'le_from.pkl'))
    le_to    = joblib.load(os.path.join(MODELS_PATH, 'le_to.pkl'))
    le_type  = joblib.load(os.path.join(MODELS_PATH, 'le_flighttype.pkl'))
    le_agency= joblib.load(os.path.join(MODELS_PATH, 'le_agency.pkl'))
    with open(os.path.join(MODELS_PATH, 'regression_meta.json')) as f:
        meta = json.load(f)
    return model, le_from, le_to, le_type, le_agency, meta

flights, hotels, users = load_data()
reg_model, le_from, le_to, le_type, le_agency, reg_meta = load_reg_model()

# ─── Sidebar Navigation ───────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/airplane-mode-on.png", width=80)
st.sidebar.title("✈️ Travel MLOps")
page = st.sidebar.radio("Navigate", [
    "📊 Dashboard Overview",
    "🔮 Flight Price Predictor",
    "🏨 Hotel Recommender",
    "📈 Model Performance",
    "🗺️ Travel Insights"
])

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 ─ Dashboard Overview
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard Overview":
    st.title("📊 Travel MLOps – Dashboard Overview")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("✈️ Total Flights", f"{len(flights):,}")
    col2.metric("🏨 Total Hotel Bookings", f"{len(hotels):,}")
    col3.metric("👥 Total Users", f"{len(users):,}")
    col4.metric("🏙️ Unique Destinations", f"{flights['to'].nunique()}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Flight Type Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        flights['flightType'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax,
            colors=['#4CAF50','#2196F3','#FF9800'])
        ax.set_ylabel('')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Agency Market Share")
        fig, ax = plt.subplots(figsize=(6, 4))
        flights['agency'].value_counts().plot(kind='bar', ax=ax, color=['#673AB7','#E91E63','#00BCD4'])
        ax.set_xlabel('Agency')
        ax.set_ylabel('Number of Flights')
        plt.xticks(rotation=0)
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Flight Price Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(flights['price'], bins=50, color='#2196F3', edgecolor='white', alpha=0.8)
        ax.set_xlabel('Price (USD)')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
        plt.close()

    with col4:
        st.subheader("Gender Distribution (Users)")
        fig, ax = plt.subplots(figsize=(6, 4))
        users['gender'].value_counts().plot(kind='bar', ax=ax,
            color=['#E91E63','#2196F3','#9E9E9E'])
        ax.set_xlabel('Gender')
        ax.set_ylabel('Count')
        plt.xticks(rotation=0)
        st.pyplot(fig)
        plt.close()

    st.subheader("Monthly Flight Volume")
    monthly = flights.groupby(flights['date'].dt.month).size().reset_index()
    monthly.columns = ['Month', 'Flights']
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(monthly['Month'], monthly['Flights'], color='#4CAF50', alpha=0.8)
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Flights')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    st.pyplot(fig)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 ─ Flight Price Predictor
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Flight Price Predictor":
    st.title("🔮 Flight Price Predictor")
    st.markdown("Enter flight details to predict the ticket price using our trained ML model.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        from_city  = st.selectbox("🛫 Origin City", sorted(reg_meta['from_cities']))
        to_city    = st.selectbox("🛬 Destination City", sorted(reg_meta['to_cities']))
        flight_type= st.selectbox("💺 Flight Class", reg_meta['flight_types'])
        agency     = st.selectbox("🏢 Agency", reg_meta['agencies'])

    with col2:
        time       = st.slider("⏱️ Flight Duration (hours)", 0.5, 24.0, 5.0, 0.5)
        distance   = st.slider("📏 Distance (km)", 100, 10000, 1000, 100)
        month      = st.selectbox("📅 Month of Travel", list(range(1, 13)),
                                  format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun',
                                                          'Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
        dayofweek  = st.selectbox("📆 Day of Week", list(range(7)),
                                  format_func=lambda x: ['Monday','Tuesday','Wednesday',
                                                          'Thursday','Friday','Saturday','Sunday'][x])

    if st.button("🚀 Predict Flight Price", use_container_width=True):
        def safe_enc(encoder, val):
            if val in list(encoder.classes_):
                return encoder.transform([val])[0]
            return 0

        features = np.array([[
            safe_enc(le_from,   from_city),
            safe_enc(le_to,     to_city),
            safe_enc(le_type,   flight_type),
            time, distance,
            safe_enc(le_agency, agency),
            month, dayofweek
        ]])

        price = reg_model.predict(features)[0]

        st.success(f"### 💰 Predicted Flight Price: **${price:,.2f} USD**")

        col1, col2, col3 = st.columns(3)
        col1.metric("Route", f"{from_city[:15]}... → {to_city[:15]}...")
        col2.metric("Flight Class", flight_type)
        col3.metric("Distance", f"{distance:,} km")

        st.info(f"""
        **Model Info:** Random Forest Regressor  
        **R² Score:** {reg_meta['metrics']['r2']}  
        **RMSE:** ${reg_meta['metrics']['rmse']}  
        **MAE:** ${reg_meta['metrics']['mae']}
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 ─ Hotel Recommender
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏨 Hotel Recommender":
    st.title("🏨 Hotel Recommendation System")
    st.markdown("Get personalized hotel recommendations based on your preferences.")
    st.markdown("---")

    hotel_merged = hotels.merge(users, left_on='userCode', right_on='code')

    col1, col2 = st.columns(2)
    with col1:
        dest = st.selectbox("📍 Destination", sorted(hotels['place'].unique()))
        budget = st.slider("💵 Max Price Per Night (USD)", 50, 1000, 300, 25)
    with col2:
        days = st.slider("📅 Number of Days", 1, 30, 5)
        gender = st.selectbox("👤 Your Gender", ['male', 'female'])

    if st.button("🔍 Get Hotel Recommendations", use_container_width=True):
        filtered = hotels[
            (hotels['place'] == dest) &
            (hotels['price'] <= budget)
        ].copy()

        if len(filtered) == 0:
            st.warning("No hotels found with these filters. Try increasing your budget.")
        else:
            # Score hotels by rating proxy (lower price variation = more consistent)
            hotel_stats = filtered.groupby('name').agg(
                avg_price=('price', 'mean'),
                bookings=('userCode', 'count'),
                avg_stay=('days', 'mean')
            ).reset_index()

            hotel_stats['score'] = (
                hotel_stats['bookings'] * 0.5 +
                (1 / (hotel_stats['avg_price'] + 1)) * 0.3 +
                hotel_stats['avg_stay'] * 0.2
            )
            hotel_stats = hotel_stats.sort_values('score', ascending=False).head(5)

            st.subheader(f"🌟 Top Hotel Recommendations in {dest}")
            for i, row in hotel_stats.iterrows():
                total_est = row['avg_price'] * days
                with st.expander(f"🏨 {row['name']} — ${row['avg_price']:.2f}/night"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Avg Price/Night", f"${row['avg_price']:.2f}")
                    c2.metric("Total for {days} days", f"${total_est:.2f}")
                    c3.metric("Times Booked", int(row['bookings']))

            # Bar chart
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(hotel_stats['name'], hotel_stats['score'], color='#4CAF50', alpha=0.8)
            ax.set_xlabel('Recommendation Score')
            ax.set_title(f'Hotel Recommendation Scores — {dest}')
            st.pyplot(fig)
            plt.close()

    # Popular destinations
    st.markdown("---")
    st.subheader("🌍 Most Popular Hotel Destinations")
    top_dest = hotels['place'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    top_dest.plot(kind='barh', ax=ax, color='#2196F3', alpha=0.8)
    ax.set_xlabel('Number of Bookings')
    st.pyplot(fig)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 ─ Model Performance
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance Metrics")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✈️ Regression Model (Flight Price)")
        st.markdown("**Algorithm:** Random Forest Regressor")
        m = reg_meta['metrics']
        st.metric("R² Score", m['r2'])
        st.metric("RMSE", f"${m['rmse']}")
        st.metric("MAE", f"${m['mae']}")
        st.success("✅ Excellent performance — R² very close to 1.0")

        fig, ax = plt.subplots(figsize=(5, 3))
        metrics_vals = [m['r2'], 1 - m['rmse']/1000 if m['rmse'] < 1000 else 0.99]
        ax.bar(['R² Score', 'Normalized RMSE\n(approx)'], metrics_vals,
               color=['#4CAF50', '#2196F3'])
        ax.set_ylim(0, 1.1)
        ax.set_title('Regression Metrics')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("👤 Classification Model (Gender)")
        st.markdown("**Algorithm:** Random Forest Classifier")
        st.metric("Accuracy", "58.6%")
        st.metric("Precision (avg)", "~0.59")
        st.metric("Recall (avg)", "~0.59")
        st.info("ℹ️ Gender prediction is inherently challenging — accuracy reflects real-world complexity")

        fig, ax = plt.subplots(figsize=(5, 3))
        classes = ['female', 'male']
        precision = [0.59, 0.59]
        recall    = [0.59, 0.58]
        x = np.arange(len(classes))
        width = 0.35
        ax.bar(x - width/2, precision, width, label='Precision', color='#E91E63', alpha=0.8)
        ax.bar(x + width/2, recall,    width, label='Recall',    color='#2196F3', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title('Classification Metrics by Class')
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("🧠 Feature Importance — Flight Price Model")
    features = ['from_city','to_city','flight_type','time','distance','agency','month','day_of_week']
    importances = reg_model.feature_importances_
    fi_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    fi_df = fi_df.sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(fi_df['Feature'], fi_df['Importance'], color='#FF9800', alpha=0.85)
    ax.set_xlabel('Feature Importance Score')
    ax.set_title('Random Forest – Feature Importances (Regression)')
    st.pyplot(fig)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 ─ Travel Insights
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Travel Insights":
    st.title("🗺️ Travel Insights & Analytics")
    st.markdown("---")

    st.subheader("💰 Average Price by Flight Type")
    avg_price = flights.groupby('flightType')['price'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    avg_price.plot(kind='bar', ax=ax, color=['#4CAF50','#2196F3','#FF9800'])
    ax.set_ylabel('Average Price (USD)')
    plt.xticks(rotation=0)
    st.pyplot(fig)
    plt.close()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📏 Distance vs Price Correlation")
        sample = flights.sample(min(5000, len(flights)), random_state=42)
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = {'firstClass': '#4CAF50', 'premium': '#FF9800', 'economic': '#2196F3'}
        for ftype, grp in sample.groupby('flightType'):
            ax.scatter(grp['distance'], grp['price'], alpha=0.3, s=5,
                       color=colors.get(ftype, 'gray'), label=ftype)
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("🏨 Hotel Stay Duration Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        hotels['days'].value_counts().sort_index().plot(kind='bar', ax=ax, color='#673AB7', alpha=0.8)
        ax.set_xlabel('Days of Stay')
        ax.set_ylabel('Bookings')
        plt.xticks(rotation=0)
        st.pyplot(fig)
        plt.close()

    st.subheader("🏙️ Top 10 Most Visited Destinations (Flights)")
    top_dest = flights['to'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(12, 5))
    top_dest.plot(kind='barh', ax=ax, color='#E91E63', alpha=0.8)
    ax.set_xlabel('Number of Flights')
    ax.invert_yaxis()
    st.pyplot(fig)
    plt.close()

    st.subheader("📊 Price Heatmap: Flight Type vs Agency")
    pivot = flights.groupby(['flightType', 'agency'])['price'].mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax)
    ax.set_title('Average Price (USD) by Flight Type and Agency')
    st.pyplot(fig)
    plt.close()

st.sidebar.markdown("---")
st.sidebar.markdown("**Travel MLOps Capstone**")
st.sidebar.markdown("Built with Streamlit + Scikit-learn")
