import pandas as pd

MODEL_INPUT_COLUMNS = [
    "Age", "Gender", "Weight (kg)", "Height (m)",
    "Session_Duration (hours)", "Workout_Type",
    "Experience_Level", "Workout_Frequency (days/week)",
]


def find_missing_features(features):
    return [k for k, v in features.items() if v is None]


def build_model_input(features, encoders):
    gender_encoded = encoders["Gender"].transform([features["Gender"]])[0]
    workout_encoded = encoders["Workout_Type"].transform([features["Workout_Type"]])[0]

    return pd.DataFrame([[
        features["Age"],
        gender_encoded,
        features["Weight_kg"],
        features["Height_m"],
        features["Session_Duration_hours"],
        workout_encoded,
        features["Experience_Level"],
        features["Workout_Frequency_days_per_week"],
    ]], columns=MODEL_INPUT_COLUMNS)
