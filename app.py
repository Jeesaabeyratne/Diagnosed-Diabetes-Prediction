import streamlit as st
import requests
import json

from numpy.ma.core import minimum_fill_value


def Diabetes_Diagnosed_Prediction():
    st.title("Diabetes Diagnosed Prediction")

    # -------------------------------
    # Numerical Inputs
    # -------------------------------
    age = st.number_input("Age", min_value=0, max_value=120, step=1, key="age")
    alcohol_consumption_per_week = st.number_input("Alcohol Consumption per Week (units)", min_value=0, step=1, key="alcohol")
    physical_activity_minutes_per_week = st.number_input("Physical Activity (minutes per week)", min_value=0, step=10, key="activity")
    diet_score = st.number_input("Diet Score", min_value=0.0, max_value=10.0, step=0.1, key="diet")
    sleep_hours_per_day = st.number_input("Sleep Hours per Day", min_value=0.0, max_value=24.0, step=0.1, key="sleep")
    screen_time_hours_per_day = st.number_input("Screen Time (hours per day)", min_value=0.0, max_value=24.0, step=0.1, key="screen")
    bmi = st.number_input("BMI", min_value=0.0, step=0.1, key="bmi")
    waist_to_hip_ratio = st.number_input("Waist to Hip Ratio", min_value=0.0, step=0.01, key="waist_hip")
    systolic_bp = st.number_input("Systolic Blood Pressure", min_value=0, step=1, key="systolic")
    diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=0, step=1, key="diastolic")
    heart_rate = st.number_input("Heart Rate", min_value=0, step=1, key="heart")
    cholesterol_total = st.number_input("Total Cholesterol", min_value=0, step=1, key="chol_total")
    hdl_cholesterol = st.number_input("HDL Cholesterol", min_value=0, step=1, key="hdl")
    ldl_cholesterol = st.number_input("LDL Cholesterol", min_value=0, step=1, key="ldl")
    triglycerides = st.number_input("Triglycerides", min_value=0, step=1, key="triglycerides")
    family_history_diabetes = st.number_input("Family History Diabetes", min_value=0, step=1, key="family_diabetes")
    hypertension_history = st.number_input("Hypertension History", min_value=0, step=1, key="hypertension")
    cardiovascular_history = st.number_input("Cardiovascular History", min_value=0, step=1, key="cardio")

    # -------------------------------
    # Categorical Inputs
    # -------------------------------
    gender_map = {"Male": 0, "Female": 1}
    ethnicity_map = {"White": 0, "Black": 1, "Asian": 2, "Hispanic": 3, "Other": 4}
    education_map = {"Primary": 0, "Secondary": 1, "Graduate": 2, "Postgraduate": 3}
    income_map = {"Low": 0, "Middle": 1, "High": 2}
    smoking_map = {"Never": 0, "Former": 1, "Current": 2}
    employment_map = {"Employed": 0, "Unemployed": 1, "Retired": 2, "Student": 3}

    gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
    ethnicity = st.selectbox("Ethnicity", ["White", "Black", "Asian", "Hispanic", "Other"], key="ethnicity")
    education_level = st.selectbox("Education Level", ["Primary", "Secondary", "Graduate", "Postgraduate"], key="education")
    income_level = st.selectbox("Income Level", ["Low", "Middle", "High"], key="income")
    smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"], key="smoking")
    employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Retired", "Student"], key="employment")

    # -------------------------------
    # Function to call FastAPI
    # -------------------------------
    def predict_diagnosed_diabetes():
        url = "http://127.0.0.1:8000/predict/"

        data = {
            "age": age,
            "alcohol_consumption_per_week": alcohol_consumption_per_week,
            "physical_activity_minutes_per_week": physical_activity_minutes_per_week,
            "diet_score": diet_score,
            "sleep_hours_per_day": sleep_hours_per_day,
            "screen_time_hours_per_day": screen_time_hours_per_day,
            "bmi": bmi,
            "waist_to_hip_ratio": waist_to_hip_ratio,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "heart_rate": heart_rate,
            "cholesterol_total": cholesterol_total,
            "hdl_cholesterol": hdl_cholesterol,
            "ldl_cholesterol": ldl_cholesterol,
            "triglycerides": triglycerides,
            "gender": gender_map[gender],
            "ethnicity": ethnicity_map[ethnicity],
            "education_level": education_map[education_level],
            "income_level": income_map[income_level],
            "smoking_status": smoking_map[smoking_status],
            "employment_status": employment_map[employment_status],
            "family_history_diabetes": family_history_diabetes,
            "hypertension_history": hypertension_history,
            "cardiovascular_history": cardiovascular_history
        }

        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            result = response.json()
            # Get both numeric value and text
            pred_value = result.get("prediction_value", None)
            text_result = result.get("result_text", "No result returned")
            if pred_value is not None:
                return f"Prediction value: {pred_value:.2f} → {text_result}"
            else:
                return text_result
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {e}")
            return None

    # -------------------------------
    # Button to trigger prediction
    # -------------------------------
    if st.button("Predict"):
        diagnosed_diabetes = predict_diagnosed_diabetes()
        if diagnosed_diabetes is not None:
            st.success(diagnosed_diabetes)

# Define functions for other pages
def home():
    st.title("DIAGNOSED DIABETES")

    # Add a picture
    image_path = "diabetespredict.jpeg"  # Update this to your image path if necessary
    st.image(image_path, caption="Diagnosed Diabetes: Predicting Risk Through Data-Driven Intelligence", use_column_width=True)

    st.markdown("""Welcome to the Diagnosed Diabetes Prediction App! 🐚🔍

Our goal is to accurately predict whether an individual is diagnosed with diabetes based on lifestyle, 
clinical, and demographic factors. By analyzing comprehensive health-related data, 
our application uses advanced machine learning algorithms to assess diabetes risk and support early identification.
Together, this system helps improve awareness, prevention, and data-driven healthcare decision-making.

How It Works

Provide Health Information:
Navigate to the Diagnosed Diabetes Prediction page and enter the required details, including age, lifestyle habits, physical measurements, 
and medical history such as BMI, blood pressure, cholesterol levels, and family history.

Submit Data:
After completing all the input fields, click the “Predict” button.
The system processes the data using a trained machine learning model optimized for diabetes prediction.

Receive Prediction:
Instantly view the prediction indicating whether the individual is likely to be diagnosed with diabetes.
The result is generated using learned patterns from historical health data and validated statistical techniques.
        """)


def about():
    st.title("About Diabetes & Dataset")

    st.write("""
    ### About Diabetes
    Diabetes is a chronic medical condition that affects how the body regulates blood sugar (glucose) levels. 
    People with diabetes either cannot produce enough insulin or their bodies cannot effectively use the insulin produced. 
    Over time, uncontrolled diabetes can lead to serious complications including heart disease, kidney failure, vision problems, and nerve damage.

    Lifestyle factors, genetics, and other health conditions can all contribute to the risk of developing diabetes. 
    Early detection and lifestyle interventions can help manage and even prevent the progression of the disease.
    """)

    image_path = "diabetespredict.jpeg"
    st.image(image_path, caption="Diabetes Awareness", use_column_width=True)

    st.write("""
    ### About the Dataset
    The Diabetes Prediction dataset is used to train machine learning models to predict an individual's risk of developing diabetes. 
    It includes demographic, lifestyle, and health-related features, making it suitable for regression or classification tasks.
    """)

    st.write("""
    The dataset includes the following features for each individual:

    1. **age**: Age of the individual in years  
    2. **alcohol_consumption_per_week**: Average number of alcoholic drinks consumed per week  
    3. **physical_activity_minutes_per_week**: Total minutes of physical activity per week  
    4. **diet_score**: A numerical score representing diet quality  
    5. **sleep_hours_per_day**: Average hours of sleep per day  
    6. **screen_time_hours_per_day**: Average hours spent on screens per day  
    7. **bmi**: Body Mass Index (weight in kg / height in m²)  
    8. **waist_to_hip_ratio**: Ratio of waist circumference to hip circumference  
    9. **systolic_bp**: Systolic blood pressure (mmHg)  
    10. **diastolic_bp**: Diastolic blood pressure (mmHg)  
    11. **heart_rate**: Resting heart rate (beats per minute)  
    12. **cholesterol_total**: Total cholesterol level (mg/dL)  
    13. **hdl_cholesterol**: High-density lipoprotein cholesterol (mg/dL)  
    14. **ldl_cholesterol**: Low-density lipoprotein cholesterol (mg/dL)  
    15. **triglycerides**: Triglyceride levels (mg/dL)  
    16. **gender**: Categorical variable (Male/Female/Other)  
    17. **ethnicity**: Ethnic background of the individual  
    18. **education_level**: Highest education level achieved  
    19. **income_level**: Annual income category  
    20. **smoking_status**: Smoking behavior (Current/Former/Never)  
    21. **employment_status**: Employment status  
    22. **family_history_diabetes**: Indicates if there is a family history of diabetes  
    23. **hypertension_history**: History of high blood pressure  
    24. **cardiovascular_history**: History of cardiovascular diseases
    """)

    st.write("""
    The target variable in this dataset is whether an individual has diabetes or is at risk of developing it. 
    Using these features, machine learning models can help predict diabetes risk and assist in early intervention and preventive healthcare.
    """)


# Create a sidebar with navigation options
st.sidebar.title("Navigation")
option = st.sidebar.selectbox(
    "Choose a page",
    ["Home", "About Diabetes and Dataset", "Diagnosed Diabetes Prediction"]
)

# Render the selected page
if option == "Home":
    home()
elif option == "About Diabetes and Dataset":
    about()
elif option == "Diagnosed Diabetes Prediction":
    (Diabetes_Diagnosed_Prediction())

# streamlit run app.py
