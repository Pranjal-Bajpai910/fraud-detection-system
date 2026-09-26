import streamlit as st
import joblib
import math
import datetime
import pandas as pd
from scipy.sparse import hstack
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROFESSIONAL UI STYLING
# =========================================================

st.markdown("""
<style>

    /* ---------- MAIN APP ---------- */

    .stApp {
        background: linear-gradient(
            135deg,
            #f8fafc 0%,
            #eef2ff 50%,
            #f8fafc 100%
        );
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ---------- HEADER ---------- */

    .main-title {
        font-size: 3.2rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 0.2rem;
        letter-spacing: -1px;
    }

    .subtitle {
        font-size: 1.15rem;
        color: #6b7280;
        margin-bottom: 1.8rem;
    }


    /* ---------- STATUS CARD ---------- */

    .status-card {
        background: white;
        padding: 1rem 1.4rem;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05);
        margin-bottom: 2rem;
    }


    /* ---------- SECTION CARD ---------- */

    .section-card {
        background: white;
        padding: 1.5rem;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.06);
        margin-bottom: 1.5rem;
    }


    /* ---------- INPUT LABELS ---------- */

    label {
        font-weight: 600 !important;
        color: #374151 !important;
    }


    /* ---------- BUTTON ---------- */

    .stButton > button {
        width: 100%;
        height: 3.4rem;
        border-radius: 12px;
        border: none;
        font-size: 1.1rem;
        font-weight: 700;
        color: white;
        background: linear-gradient(
            90deg,
            #4f46e5,
            #7c3aed
        );
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(79, 70, 229, 0.30);
    }


    /* ---------- PREDICTION CARD ---------- */

    .prediction-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 12px 35px rgba(15, 23, 42, 0.08);
        text-align: center;
        margin-top: 2rem;
    }

    .prediction-title {
        font-size: 1rem;
        font-weight: 700;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    .probability {
        font-size: 3.5rem;
        font-weight: 800;
        color: #111827;
        margin: 0.5rem 0;
    }


    /* ---------- SIDEBAR ---------- */

    [data-testid="stSidebar"] {
        background: #111827;
    }

    [data-testid="stSidebar"] * {
        color: #f9fafb;
    }

    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #d1d5db;
    }

    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: white;
    }


    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e5e7eb;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL AND PREPROCESSING OBJECTS
# =========================================================

model = joblib.load(
    BASE_DIR / "fraud_logistic_cleanlog_model.pkl"
)

scaler = joblib.load(
    BASE_DIR / "scaler_cleanlog.pkl"
)

encoder = joblib.load(
    BASE_DIR / "encoder.pkl"
)

features = joblib.load(
    BASE_DIR / "notebooks" / "feature_columns.pkl"
)

with open(
    BASE_DIR / "final_threshold_cleanlog.txt",
    "r"
) as f:
    threshold = float(f.read())
    
# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🛡️ FraudGuard</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Machine Learning Transaction Risk Detection"
    )

    st.markdown("---")

    st.markdown("### 📊 Model Performance")

    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Model</div>
        <div class="metric-value">Logistic Regression</div>
    </div>

    <div class="metric-card">
        <div class="metric-label">ROC-AUC</div>
        <div class="metric-value">0.902</div>
    </div>

    <div class="metric-card">
        <div class="metric-label">PR-AUC</div>
        <div class="metric-value">0.479</div>
    </div>

    <div class="metric-card">
        <div class="metric-label">F1 Score</div>
        <div class="metric-value">0.547</div>
    </div>

    <div class="metric-card">
        <div class="metric-label">Detection Recall</div>
        <div class="metric-value">51.17%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### ⚙️ Preprocessing")

    st.write("✓ StandardScaler")
    st.write("✓ OneHotEncoder")
    st.write("✓ Cyclical Time Features")
    st.write("✓ Geographical Distance")
    st.write("✓ Optimized Classification Threshold")
    st.write("✓ Log-Transformed Transaction Amount")

    st.markdown("---")

    st.caption(
        "Built as an end-to-end Data Science / ML portfolio project."
    )


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🛡️ Fraud Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered transaction risk assessment using machine learning'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# MODEL STATUS
# =========================================================

st.markdown("""
<div class="status-card">
    🟢 <b>System Ready</b>
    &nbsp; • &nbsp;
   Clean Log Logistic Regression loaded
    &nbsp; • &nbsp;
    2,167 processed features
    &nbsp; • &nbsp;
    F1-optimized threshold
</div>
""", unsafe_allow_html=True)


# =========================================================
# TRANSACTION DETAILS
# =========================================================

st.markdown("## 💳 Transaction Details")


col1, col2 = st.columns(2)


with col1:

    amt = st.number_input(
        "Transaction Amount ($)",
        min_value=0.0,
        value=100.0,
        step=1.0
    )

    category = st.selectbox(
        "Transaction Category",
        encoder.categories_[1]
    )

    merchant = st.selectbox(
        "Merchant",
        encoder.categories_[0]
    )

    gender = st.selectbox(
        "Gender",
        ["F", "M"]
    )


with col2:

    city = st.selectbox(
        "City",
        encoder.categories_[3]
    )

    state = st.selectbox(
        "State",
        encoder.categories_[4]
    )

    job = st.selectbox(
        "Job",
        encoder.categories_[5]
    )


# =========================================================
# CUSTOMER AND MERCHANT INFORMATION
# =========================================================

st.markdown("## 📍 Customer & Merchant Information")


col1, col2 = st.columns(2)


with col1:

    st.markdown("### 👤 Customer")

    lat = st.number_input(
        "Customer Latitude",
        min_value=-90.0,
        max_value=90.0,
        value=40.0
    )

    long = st.number_input(
        "Customer Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=-75.0
    )

    city_pop = st.number_input(
        "City Population",
        min_value=0,
        value=50000
    )

    customer_age = st.number_input(
        "Customer Age",
        min_value=18,
        max_value=100,
        value=50
    )

    card_transaction_count = st.number_input(
        "Card Transaction Count",
        min_value=0,
        value=100
    )


with col2:

    st.markdown("### 🏪 Merchant")

    merch_lat = st.number_input(
        "Merchant Latitude",
        min_value=-90.0,
        max_value=90.0,
        value=40.0
    )

    merch_long = st.number_input(
        "Merchant Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=-75.0
    )


# =========================================================
# TRANSACTION TIME
# =========================================================

st.markdown("## 🕒 Transaction Time")


col1, col2 = st.columns(2)


with col1:

    transaction_date = st.date_input(
        "Transaction Date"
    )


with col2:

    transaction_time = st.time_input(
        "Transaction Time"
    )


# =========================================================
# FEATURE ENGINEERING
# =========================================================

transaction_datetime = datetime.datetime.combine(
    transaction_date,
    transaction_time
)

unix_time = int(
    transaction_datetime.timestamp()
)

transaction_year = transaction_date.year
hour = transaction_time.hour
day = transaction_date.day
month = transaction_date.month
weekday = transaction_date.weekday()


# ---------- Cyclical Time Features ----------

hour_sin = math.sin(
    2 * math.pi * hour / 24
)

hour_cos = math.cos(
    2 * math.pi * hour / 24
)

day_sin = math.sin(
    2 * math.pi * day / 31
)

day_cos = math.cos(
    2 * math.pi * day / 31
)

month_sin = math.sin(
    2 * math.pi * month / 12
)

month_cos = math.cos(
    2 * math.pi * month / 12
)

weekday_sin = math.sin(
    2 * math.pi * weekday / 7
)

weekday_cos = math.cos(
    2 * math.pi * weekday / 7
)


# =========================================================
# HAVERSINE DISTANCE
# =========================================================

lat1 = math.radians(lat)
lon1 = math.radians(long)

lat2 = math.radians(merch_lat)
lon2 = math.radians(merch_long)

dlat = lat2 - lat1
dlon = lon2 - lon1

a = (
    math.sin(dlat / 2) ** 2
    + math.cos(lat1)
    * math.cos(lat2)
    * math.sin(dlon / 2) ** 2
)

c = 2 * math.atan2(
    math.sqrt(a),
    math.sqrt(1 - a)
)

distance_km = 6371 * c


# =========================================================
# CREATE INPUT DATAFRAME
# =========================================================

input_data = pd.DataFrame([{

    "amt": amt,

    "lat": lat,

    "long": long,

    "city_pop": city_pop,

    "unix_time": unix_time,

    "merch_lat": merch_lat,

    "merch_long": merch_long,

    "distance_km": distance_km,

    "transaction_year": transaction_year,

    "customer_age": customer_age,

    "card_transaction_count": card_transaction_count,

    "hour_sin": hour_sin,

    "hour_cos": hour_cos,

    "day_sin": day_sin,

    "day_cos": day_cos,

    "month_sin": month_sin,

    "month_cos": month_cos,

    "weekday_sin": weekday_sin,

    "weekday_cos": weekday_cos,

    "merchant": merchant,

    "category": category,

    "gender": gender,

    "city": city,

    "state": state,

    "job": job

}])


# =========================================================
# PREPROCESSING
# =========================================================

numerical_cols = features["numerical_cols"]

categorical_cols = features["categorical_cols"]


input_num = input_data[numerical_cols].copy()

input_cat = input_data[categorical_cols]


# Match the training-time amount transformation
input_num["amt"] = input_num["amt"].map(math.log1p)

input_num_scaled = scaler.transform(
    input_num
)

input_cat_encoded = encoder.transform(
    input_cat
)


# =========================================================
# COMBINE FEATURES
# =========================================================

input_processed = hstack([
    input_num_scaled,
    input_cat_encoded
])
# =========================================================
# DEPLOYMENT PIPELINE VERIFICATION
# =========================================================

expected_features = model.n_features_in_
actual_features = input_processed.shape[1]

if actual_features != expected_features:
    st.error(
        f"Pipeline mismatch: model expects {expected_features} "
        f"features, but received {actual_features}."
    )
    st.stop()


# =========================================================
# PREDICTION
# =========================================================

st.markdown("---")

st.markdown("## 🔍 Transaction Risk Analysis")


if st.button(
    "🔍 Analyze Transaction",
    type="primary"
):

    probability = model.predict_proba(
        input_processed
    )[0, 1]

    prediction = int(
        probability >= threshold
    )


    # ---------------------------------------------
    # Prediction Card
    # ---------------------------------------------

    st.markdown(
        '<div class="prediction-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="prediction-title">'
        'Risk Assessment'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="probability">'
        f'{probability:.2%}'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # ---------------------------------------------
    # Result
    # ---------------------------------------------

    if prediction == 1:

        st.error(
            "🚨 Potential Fraudulent Transaction"
        )

        st.progress(
            min(float(probability), 1.0),
            text="Fraud Risk"
        )

        st.warning(
            "This transaction has been classified "
            "as potentially fraudulent by the model."
        )

    else:

        st.success(
            "✅ Legitimate Transaction"
        )

        st.progress(
            min(float(probability), 1.0),
            text="Fraud Risk"
        )

        st.info(
            "The model currently classifies this "
            "transaction as legitimate."
        )


    # ---------------------------------------------
    # Prediction Details
    # ---------------------------------------------

    st.markdown("### 📋 Prediction Details")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:

        st.metric(
            "Fraud Probability",
            f"{probability:.2%}"
        )

    with result_col2:

        st.metric(
            "Decision Threshold",
            f"{threshold:.2%}"
        )

    with result_col3:

        if prediction == 1:

            st.metric(
                "Classification",
                "Fraud Risk"
            )

        else:

            st.metric(
                "Classification",
                "Legitimate"
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    🛡️ Fraud Detection System
    <br>
    Machine Learning • Feature Engineering •
    Imbalanced Classification • Risk Prediction
</div>
""", unsafe_allow_html=True)