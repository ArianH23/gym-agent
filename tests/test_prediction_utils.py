from agent.prediction_utils import build_model_input, find_missing_features


class FakeEncoder:
    def __init__(self, mapping):
        self.mapping = mapping

    def transform(self, values):
        return [self.mapping[v] for v in values]


def _encoders():
    return {
        "Gender": FakeEncoder({"Male": 1, "Female": 0}),
        "Workout_Type": FakeEncoder({"HIIT": 2, "Cardio": 1, "Strength": 3, "Yoga": 0}),
    }


def test_find_missing_features_returns_keys_with_none_values():
    features = {"Age": 30, "Gender": None, "Weight_kg": 80, "Height_m": None}
    result = find_missing_features(features)
    assert result == ["Gender", "Height_m"]


def test_find_missing_features_returns_empty_list_when_all_present():
    features = {"Age": 30, "Gender": "Male", "Weight_kg": 80}
    result = find_missing_features(features)
    assert result == []


def test_build_model_input_has_expected_columns():
    features = {
        "Age": 30, "Gender": "Male", "Weight_kg": 80.0, "Height_m": 1.75,
        "Session_Duration_hours": 0.75, "Workout_Type": "HIIT",
        "Experience_Level": 2, "Workout_Frequency_days_per_week": 3,
    }
    df = build_model_input(features, _encoders())
    assert list(df.columns) == [
        "Age", "Gender", "Weight (kg)", "Height (m)",
        "Session_Duration (hours)", "Workout_Type",
        "Experience_Level", "Workout_Frequency (days/week)",
    ]


def test_build_model_input_encodes_gender_and_workout_type():
    features = {
        "Age": 30, "Gender": "Female", "Weight_kg": 60.0, "Height_m": 1.6,
        "Session_Duration_hours": 1.0, "Workout_Type": "Cardio",
        "Experience_Level": 1, "Workout_Frequency_days_per_week": 4,
    }
    df = build_model_input(features, _encoders())
    row = df.iloc[0]
    assert row["Gender"] == 0
    assert row["Workout_Type"] == 1


def test_build_model_input_preserves_numeric_values():
    features = {
        "Age": 42, "Gender": "Male", "Weight_kg": 90.5, "Height_m": 1.8,
        "Session_Duration_hours": 0.5, "Workout_Type": "Strength",
        "Experience_Level": 3, "Workout_Frequency_days_per_week": 5,
    }
    df = build_model_input(features, _encoders())
    row = df.iloc[0]
    assert row["Age"] == 42
    assert row["Weight (kg)"] == 90.5
    assert row["Height (m)"] == 1.8
    assert row["Workout_Frequency (days/week)"] == 5
