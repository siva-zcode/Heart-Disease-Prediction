import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path
from PIL import Image
import joblib
import json

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Healthcare Analytics Dashboard", layout="wide")
st.title("🏥 Healthcare Analytics Dashboard: SQL + EDA + ML")

# -----------------------------
# Connect SQLite DB
# -----------------------------
conn = sqlite3.connect("database/healthcare.db")

# -----------------------------
# SQL Analysis Section (18 Queries)
# -----------------------------
st.header("📊 SQL Analysis Insights")

queries = {
    "1. Total Patients": "SELECT COUNT(*) AS total_patients FROM patients",

    "2. Disease Distribution": """
        SELECT 
    CASE 
        WHEN target = 0 THEN 'No Heart Disease'
        WHEN target = 1 THEN 'Heart Disease'
    END AS heart_disease_status,
    COUNT(*) AS count
FROM patients
GROUP BY target
    """,

    "3. Average Age": "SELECT ROUND(AVG(age),2) AS avg_age FROM patients",

    "4. Patients by Sex": """
        SELECT 
    CASE 
        WHEN sex = 0 THEN 'Male'
        WHEN sex = 1 THEN 'Female'
    END AS gender,
    COUNT(*) AS count
FROM patients
GROUP BY sex
    """,

    "5. High Cholesterol Patients (>250)": """
        SELECT 
    age,
    CASE 
        WHEN sex = 0 THEN 'Male'
        WHEN sex = 1 THEN 'Female'
    END AS gender,
    cholesterol,
    CASE
        WHEN target = 0 THEN 'No Heart Disease'
        WHEN target = 1 THEN 'Heart Disease'
    END AS heart_disease_status
FROM patients
WHERE cholesterol > 250
ORDER BY cholesterol DESC
LIMIT 10
    """,

    "6. Disease by Age Group": """
        SELECT age_group, COUNT(*) AS total,
               SUM(target) AS disease_cases
        FROM patients
        GROUP BY age_group
    """,

    "7. Avg Cholesterol by Disease": """
        SELECT 
    CASE 
        WHEN target = 0 THEN 'No Heart Disease'
        WHEN target = 1 THEN 'Heart Disease'
    END AS heart_disease_status,
    ROUND(AVG(cholesterol), 2) AS avg_chol
FROM patients
GROUP BY target

    """,

    "8. Patients with Exercise Angina": """
        SELECT COUNT(*) AS angina_count
        FROM patients
        WHERE exercise_angina = 1
    """,

    "9. High BP Patients (>140)": """
        SELECT COUNT(*) AS high_bp
        FROM patients
        WHERE resting_bp_s > 140
    """,

    "10. Avg Max Heart Rate": """
        SELECT ROUND(AVG(max_heart_rate),2) AS avg_max_hr
        FROM patients
    """,

    "11. Disease by Chest Pain Type": """
        SELECT 
    CASE 
        WHEN chest_pain_type = 1 THEN 'Typical Angina'
        WHEN chest_pain_type = 2 THEN 'Atypical Angina'
        WHEN chest_pain_type = 3 THEN 'Non-Anginal Pain'
        WHEN chest_pain_type = 4 THEN 'Asymptomatic'
    END AS chest_pain,
    COUNT(*) AS total,
    SUM(target) AS disease_cases
FROM patients
GROUP BY chest_pain_type
    """,

    "12. Oldpeak Risk Grouping": """
        SELECT 
          CASE 
            WHEN oldpeak < 1 THEN 'Low'
            WHEN oldpeak BETWEEN 1 AND 3 THEN 'Medium'
            ELSE 'High'
          END AS oldpeak_group,
          COUNT(*) AS total
        FROM patients
        GROUP BY oldpeak_group
    """,

    "13. Resting ECG Distribution": """
        SELECT 
    CASE
        WHEN resting_ecg = 0 THEN 'Normal'
        WHEN resting_ecg = 1 THEN 'ST-T Wave Abnormality'
        WHEN resting_ecg = 2 THEN 'Left Ventricular Hypertrophy'
    END AS resting_ecg_status,
    COUNT(*) AS count
FROM patients
GROUP BY resting_ecg
    """,

    "14. Fasting Blood Sugar >120": """
        SELECT COUNT(*) AS fbs_high
        FROM patients
        WHERE fasting_blood_sugar = 1
    """,

    "15. Avg BP by Disease": """
        SELECT 
    CASE 
        WHEN target = 0 THEN 'No Heart Disease'
        WHEN target = 1 THEN 'Heart Disease'
    END AS heart_disease_status,
    ROUND(AVG(resting_bp_s), 2) AS avg_bp
FROM patients
GROUP BY target
    """,

    "16. Slope vs Disease": """
        SELECT 
    CASE
        WHEN st_slope = 0 THEN 'Upsloping'
        WHEN st_slope = 1 THEN 'Flat'
        WHEN st_slope = 2 THEN 'Downsloping'
        WHEN st_slope = 3 THEN 'Unknown'
    END AS st_slope_type,
    COUNT(*) AS total,
    SUM(target) AS disease_cases
FROM patients
GROUP BY st_slope
    """,

    "17. Top 10 Oldest Patients": """
        SELECT 
    age,
    CASE 
        WHEN sex = 0 THEN 'Male'
        WHEN sex = 1 THEN 'Female'
    END AS gender,
    cholesterol,
    CASE
        WHEN target = 0 THEN 'No Heart Disease'
        WHEN target = 1 THEN 'Heart Disease'
    END AS heart_disease_status
FROM patients
ORDER BY age DESC
LIMIT 10
    """,

    "18. Healthy Patients (No Disease)": """
        SELECT COUNT(*) AS healthy_count
        FROM patients
        WHERE target = 0
    """
}

for title, query in queries.items():
    st.subheader(title)
    df = pd.read_sql_query(query, conn)

    if df.shape == (1, 1):
        st.metric(title, int(df.iloc[0, 0]))
    else:
        st.dataframe(df, use_container_width=True)


# -----------------------------
# Display EDA Images
# -----------------------------
st.header("📈 Exploratory Data Analysis (EDA)")

images_path = Path("images")
eda_images = [
    ("age_vs_disease.png", "Age vs Disease"),
    ("chol_vs_disease.png", "Cholesterol vs Disease"),
    ("correlation_heatmap.png", "Correlation Heatmap"),
    ("disease_distribution.png", "Disease Distribution")
]

for img_file, caption in eda_images:
    img_path = images_path / img_file
    if img_path.exists():
        st.subheader(caption)
        st.image(Image.open(img_path), use_container_width=True)
    else:
        st.warning(f"Missing image: {img_file}")

# -----------------------------
# ML Predictions Section
# -----------------------------
st.header("🤖 ML Predicted Disease Risk")

pred_df = pd.read_csv("predicted_disease_risk.csv")
risk_counts = pred_df["Risk_Category"].value_counts()

col1, col2, col3 = st.columns(3)
col1.metric("Low Risk", int(risk_counts.get("Low", 0)))
col2.metric("Medium Risk", int(risk_counts.get("Medium", 0)))
col3.metric("High Risk", int(risk_counts.get("High", 0)))

st.subheader("Prediction Sample")
st.dataframe(pred_df.head(20), use_container_width=True)

# -----------------------------
# Predict Your Own Risk
# -----------------------------
st.header("🧍 Predict Your Own Disease Risk")

best_model = joblib.load("models/final_model.pkl")
scaler = joblib.load("models/scaler.pkl")

with open("models/feature_columns.json") as f:
    feature_columns = json.load(f)

age = st.number_input("Age", min_value=0, max_value=120, value=50)
sex = st.selectbox("Sex", ["0 - Female", "1 - Male"])
chest_pain_type = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
resting_bp_s = st.number_input("Resting Blood Pressure (mmHg)", 80, 250, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar >120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG Result", [0, 1, 2])
max_heart_rate = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
st_slope = st.selectbox("Slope of ST segment", [0, 1, 2])

input_df = pd.DataFrame({
    "age": [age],
    "sex": [int(sex.split(" - ")[0])],
    "chest_pain_type": [chest_pain_type],
    "resting_bp_s": [resting_bp_s],
    "cholesterol": [cholesterol],
    "fasting_blood_sugar": [fasting_blood_sugar],
    "resting_ecg": [resting_ecg],
    "max_heart_rate": [max_heart_rate],
    "exercise_angina": [exercise_angina],
    "oldpeak": [oldpeak],
    "st_slope": [st_slope]
})

input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=feature_columns, fill_value=0)
input_scaled = scaler.transform(input_df)

risk_prob = best_model.predict_proba(input_scaled)[:, 1][0]
risk_percentage = round(risk_prob * 100, 2)
risk_category = "Low" if risk_percentage < 30 else "Medium" if risk_percentage < 60 else "High"

st.subheader("📌 Predicted Risk")
st.metric("Disease Risk (%)", risk_percentage)
st.metric("Risk Category", risk_category)

conn.close()
