import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import pickle
import os

CSV_PATH = "data/raw/raw_data.csv"
MODEL_PATH = "models/calorie_predictor.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/encoders.pkl"

df = pd.read_csv(CSV_PATH)

FEATURES = ['Age', 'Gender', 'Weight (kg)', 'Height (m)',
            'Session_Duration (hours)', 'Workout_Type',
            'Experience_Level', 'Workout_Frequency (days/week)']

TARGET = 'Calories_Burned'

X = df[FEATURES].copy()
y = df[TARGET]

# Encode categoricals
encoders = {}
for col in ['Gender', 'Workout_Type']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
mlflow.set_experiment("calorie_predictor")
with mlflow.start_run():
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.4f}")

    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(model, "calorie_predictor")

# Save artifacts locally for the agent to load
os.makedirs("models", exist_ok=True)
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)
with open(ENCODER_PATH, 'wb') as f:
    pickle.dump(encoders, f)

print("Model, scaler and encoders saved.")