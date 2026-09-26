from fastapi import FastAPI, HTTPException
import joblib
import math
import pandas as pd
from scipy.sparse import hstack

from pydantic import BaseModel
from datetime import date, time, datetime


app = FastAPI(
    title="Fraud Detection API",
    description="REST API for the fraud detection system",
    version="1.0.0"
)


# Load trained model and preprocessing artifacts
model = joblib.load("fraud_logistic_cleanlog_model.pkl")
scaler = joblib.load("scaler_cleanlog.pkl")
encoder = joblib.load("encoder.pkl")
features = joblib.load("notebooks/feature_columns.pkl")

with open("final_threshold_cleanlog.txt", "r") as f:
    threshold = float(f.read())


class TransactionRequest(BaseModel):
    amt: float

    merchant: str
    category: str
    gender: str
    city: str
    state: str
    job: str

    lat: float
    long: float
    city_pop: int

    customer_age: int
    card_transaction_count: int

    merch_lat: float
    merch_long: float

    transaction_date: date
    transaction_time: time


# ADD YOUR create_features() FUNCTION HERE
def create_features(transaction: TransactionRequest):

    transaction_datetime = datetime.combine(
        transaction.transaction_date,
        transaction.transaction_time
    )

    unix_time = int(transaction_datetime.timestamp())

    transaction_year = transaction.transaction_date.year

    hour = transaction.transaction_time.hour
    day = transaction.transaction_date.day
    month = transaction.transaction_date.month
    weekday = transaction.transaction_date.weekday()

    # Cyclical time features
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)

    day_sin = math.sin(2 * math.pi * day / 31)
    day_cos = math.cos(2 * math.pi * day / 31)

    month_sin = math.sin(2 * math.pi * month / 12)
    month_cos = math.cos(2 * math.pi * month / 12)

    weekday_sin = math.sin(2 * math.pi * weekday / 7)
    weekday_cos = math.cos(2 * math.pi * weekday / 7)

    # Haversine distance
    lat1 = math.radians(transaction.lat)
    lon1 = math.radians(transaction.long)

    lat2 = math.radians(transaction.merch_lat)
    lon2 = math.radians(transaction.merch_long)

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

    return pd.DataFrame([{
        "amt": transaction.amt,
        "lat": transaction.lat,
        "long": transaction.long,
        "city_pop": transaction.city_pop,
        "unix_time": unix_time,
        "merch_lat": transaction.merch_lat,
        "merch_long": transaction.merch_long,
        "distance_km": distance_km,
        "transaction_year": transaction_year,
        "customer_age": transaction.customer_age,
        "card_transaction_count": transaction.card_transaction_count,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "day_sin": day_sin,
        "day_cos": day_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "weekday_sin": weekday_sin,
        "weekday_cos": weekday_cos,
        "merchant": transaction.merchant,
        "category": transaction.category,
        "gender": transaction.gender,
        "city": transaction.city,
        "state": transaction.state,
        "job": transaction.job
    }])

   
   
# -----------------------------
# API Endpoints
# -----------------------------


@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running",
        "model": "Clean Log Logistic Regression",
        "threshold": threshold
    }

@app.post("/predict")
def predict(transaction: TransactionRequest):

    # Create the same features used during training
    input_data = create_features(transaction)

    # Get feature column names
    numerical_cols = features["numerical_cols"]
    categorical_cols = features["categorical_cols"]

    # Separate numerical and categorical features
    input_num = input_data[numerical_cols].copy()
    input_cat = input_data[categorical_cols]

    # Apply the same transformation used during training
    input_num["amt"] = input_num["amt"].map(math.log1p)

    # Scale numerical features
    input_num_scaled = scaler.transform(input_num)

    # One-hot encode categorical features
    input_cat_encoded = encoder.transform(input_cat)

    # Combine both
    input_processed = hstack([
        input_num_scaled,
        input_cat_encoded
    ])

    # Safety check
    expected_features = model.n_features_in_
    actual_features = input_processed.shape[1]

    if actual_features != expected_features:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Pipeline mismatch: model expects {expected_features} "
                f"features, but received {actual_features}."
            )
        )

    # Get fraud probability
    probability = float(
        model.predict_proba(input_processed)[0, 1]
    )

    # Apply validation-selected threshold
    prediction = int(probability >= threshold)

    return {
        "fraud_probability": probability,
        "prediction": prediction,
        "classification": (
            "Fraud" if prediction == 1 else "Legitimate"
        )
    }