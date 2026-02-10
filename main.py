from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the best model
best_model = joblib.load('best_model.joblib')

# Load train data to get feature names for input
train_df = pd.read_csv('train (1).csv')
feature_names = train_df.columns.drop(['id', 'diagnosed_diabetes'])

# Label Encoders
gender_le = LabelEncoder()
ethnicity_le = LabelEncoder()
education_le = LabelEncoder()
income_le = LabelEncoder()
smoking_le = LabelEncoder()
employment_le = LabelEncoder()

# Fit encoders on training data
train_df["gender"] = gender_le.fit_transform(train_df["gender"])
train_df["ethnicity"] = ethnicity_le.fit_transform(train_df["ethnicity"])
train_df["education_level"] = education_le.fit_transform(train_df["education_level"])
train_df["income_level"] = income_le.fit_transform(train_df["income_level"])
train_df["smoking_status"] = smoking_le.fit_transform(train_df["smoking_status"])
train_df["employment_status"] = employment_le.fit_transform(train_df["employment_status"])

# Encode dictionaries if you want
gender_encode = dict(zip(gender_le.classes_, gender_le.transform(gender_le.classes_)))
ethnicity_encode = dict(zip(ethnicity_le.classes_, ethnicity_le.transform(ethnicity_le.classes_)))
education_encode = dict(zip(education_le.classes_, education_le.transform(education_le.classes_)))
income_encode = dict(zip(income_le.classes_, income_le.transform(income_le.classes_)))
smoking_encode = dict(zip(smoking_le.classes_, smoking_le.transform(smoking_le.classes_)))
employment_encode = dict(zip(employment_le.classes_, employment_le.transform(employment_le.classes_)))

# FastAPI app
app = FastAPI()

# Pydantic model
class InputData(BaseModel):
    age: int
    alcohol_consumption_per_week: float
    physical_activity_minutes_per_week: int
    diet_score: float
    sleep_hours_per_day: float
    screen_time_hours_per_day: float
    bmi: float
    waist_to_hip_ratio: float
    systolic_bp: int
    diastolic_bp: int
    heart_rate: int
    cholesterol_total: float
    hdl_cholesterol: float
    ldl_cholesterol: float
    triglycerides: float
    gender: int
    ethnicity: int
    education_level: int
    income_level: int
    smoking_status: int
    employment_status: int
    family_history_diabetes: int
    hypertension_history: int
    cardiovascular_history: int

# API route (outside the InputData class)
@app.post("/predict/")
async def predict(input_data: InputData):
    try:
        input_values = [[
            input_data.age,
            input_data.alcohol_consumption_per_week,
            input_data.physical_activity_minutes_per_week,
            input_data.diet_score,
            input_data.sleep_hours_per_day,
            input_data.screen_time_hours_per_day,
            input_data.bmi,
            input_data.waist_to_hip_ratio,
            input_data.systolic_bp,
            input_data.diastolic_bp,
            input_data.heart_rate,
            input_data.cholesterol_total,
            input_data.hdl_cholesterol,
            input_data.ldl_cholesterol,
            input_data.triglycerides,
            input_data.gender,
            input_data.ethnicity,
            input_data.education_level,
            input_data.income_level,
            input_data.smoking_status,
            input_data.employment_status,
            input_data.family_history_diabetes,
            input_data.hypertension_history,
            input_data.cardiovascular_history
        ]]

        # Make prediction
        prediction = best_model.predict(input_values)[0]

        # Diabetes decision logic
        if prediction < 0.5:
            result_text = "Patient is not diagnosed with diabetes"
        else:
            result_text = "Patient is likely diagnosed with diabetes"

        return {
            "prediction_value": float(prediction),
            "result_text": result_text
        }


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred: {e}")

@app.get("/")
def root():
    return {"message": "Welcome to the Diagnosed Diabetes Prediction API!"}

#uvicorn main:app --reload