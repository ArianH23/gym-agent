import json
import pickle

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.state import AgentState
from agent.text_utils import strip_code_fences

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

MODEL_PATH = "models/calorie_predictor.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/encoders.pkl"

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
with open(ENCODER_PATH, 'rb') as f:
    encoders = pickle.load(f)

EXTRACT_PROMPT = """You are a feature extractor for a calorie prediction model.
Extract the following features from the user's question and return ONLY a JSON object.
If a value is not mentioned, use null.

Features to extract:
- Age (number)
- Gender (\"Male\" or \"Female\")
- Weight_kg (number)
- Height_m (number)
- Session_Duration_hours (number)
- Workout_Type (\"Strength\", \"Yoga\", \"HIIT\", or \"Cardio\")
- Workout_Frequency_days_per_week (number)
- Experience_Level (1=beginner, 2=intermediate, 3=expert)

Return ONLY valid JSON, no explanation, no markdown."""


def predict_calories(state: AgentState) -> dict:
    question = state["messages"][-1].content
    print(f"[predict_calories] extracting features from: {question}")

    response = llm.invoke([
        SystemMessage(content=EXTRACT_PROMPT),
        HumanMessage(content=question)
    ])

    raw = strip_code_fences(response.content)

    try:
        features = json.loads(raw)
    except Exception as e:
        return {
            "predicted_calories": None,
            "dataset_results": f"Could not extract features from question: {e!s}"
        }

    print(f"[predict_calories] extracted features: {features}")

    nulls = [k for k, v in features.items() if v is None]
    if nulls:
        return {
            "predicted_calories": None,
            "ambiguous_query": True,
            "dataset_results": f"Missing required information: {', '.join(nulls)}"
        }

    try:
        gender_encoded = encoders['Gender'].transform([features['Gender']])[0]
        workout_encoded = encoders['Workout_Type'].transform([features['Workout_Type']])[0]

        input_df = pd.DataFrame([[
            features['Age'],
            gender_encoded,
            features['Weight_kg'],
            features['Height_m'],
            features['Session_Duration_hours'],
            workout_encoded,
            features['Experience_Level'],
            features['Workout_Frequency_days_per_week'],
        ]], columns=['Age', 'Gender', 'Weight (kg)', 'Height (m)',
                     'Session_Duration (hours)', 'Workout_Type',
                     'Experience_Level', 'Workout_Frequency (days/week)'])

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        print(f"[predict_calories] prediction: {prediction:.2f}")
        return {
            "predicted_calories": float(prediction),
            "prediction_inputs": features,
            "dataset_results": f"Predicted calories burned: {prediction:.2f}"
        }

    except Exception as e:
        return {
            "predicted_calories": None,
            "dataset_results": f"Prediction error: {e!s}"
        }