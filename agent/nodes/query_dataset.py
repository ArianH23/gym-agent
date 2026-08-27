import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.state import AgentState
from agent.text_utils import strip_code_fences

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

CSV_PATH = "data/raw/raw_data.csv"

COLUMN_SCHEMA = """
Age (float64), Gender (object), Weight (kg) (float64), Height (m) (float64),
Max_BPM (float64), Avg_BPM (float64), Resting_BPM (float64),
Session_Duration (hours) (float64), Calories_Burned (float64),
Workout_Type (object), Fat_Percentage (float64), Water_Intake (liters) (float64),
Workout_Frequency (days/week) (float64), Experience_Level (float64), BMI (float64),
Daily meals frequency (float64), Physical exercise (float64),
Carbs (float64), Proteins (float64), Fats (float64), Calories (float64),
meal_name (object), meal_type (object), diet_type (object),
sugar_g (float64), sodium_mg (float64), cholesterol_mg (float64),
serving_size_g (float64), cooking_method (object), prep_time_min (float64),
cook_time_min (float64), rating (float64), Name of Exercise (object),
Sets (float64), Reps (float64), Benefit (object),
Burns Calories (per 30 min) (float64), Target Muscle Group (object),
Equipment Needed (object), Difficulty Level (object), Body Part (object),
Type of Muscle (object), Workout (object), BMI_calc (float64),
cal_from_macros (float64), pct_carbs (float64), protein_per_kg (float64),
pct_HRR (float64), pct_maxHR (float64), cal_balance (float64),
lean_mass_kg (float64), expected_burn (float64),
Burns Calories (per 30 min)_bc (float64), Burns_Calories_Bin (object)
"""

QUERY_PROMPT = f"""You are a pandas code generator for a fitness dataset.
The dataframe is already loaded as `df` and has these columns:

{COLUMN_SCHEMA}

Given a user question, write ONLY the Python/pandas code to answer it.
Store the final result in a variable called `result`.
Do not import pandas. Do not load any files. Do not print anything.
Return ONLY the code, no explanations, no markdown, no backticks."""


def query_dataset(state: AgentState) -> dict:
    question = state["messages"][-1].content
    print(f"[query_dataset] generating pandas code for: {question}")

    response = llm.invoke([
        SystemMessage(content=QUERY_PROMPT),
        HumanMessage(content=question)
    ])

    code = strip_code_fences(response.content)

    print(f"[query_dataset] generated code:\n{code}")

    try:
        df = pd.read_csv(CSV_PATH)
        local_vars = {"df": df}
        exec(code, {}, local_vars)
        result = str(local_vars.get("result", "No result variable found"))
    except Exception as e:
        result = f"Error executing query: {e!s}"

    print(f"[query_dataset] result: {result}")
    return {
        "dataset_results": result,
        "pandas_query_code": code
    }