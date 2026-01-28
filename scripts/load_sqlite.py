import sqlite3
import pandas as pd

# ----------------------------
# Load cleaned CSV
# ----------------------------
df = pd.read_csv("data/processed/heart_disease_cleaned.csv")

# ----------------------------
# Connect to SQLite database
# ----------------------------
conn = sqlite3.connect("database/healthcare.db")
cursor = conn.cursor()

# ----------------------------
# Create table with proper column types
# ----------------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS patients (
    age REAL,
    sex INTEGER,
    chest_pain_type INTEGER,
    resting_bp_s REAL,
    cholesterol REAL,
    fasting_blood_sugar INTEGER,
    resting_ecg INTEGER,
    max_heart_rate REAL,
    exercise_angina INTEGER,
    oldpeak REAL,
    st_slope INTEGER,
    target INTEGER,
    age_group TEXT,
    chol_category TEXT
)
""")

# ----------------------------
# Insert data from DataFrame
# ----------------------------
# Ensure all required columns exist in CSV
expected_cols = [
    "age", "sex", "chest_pain_type", "resting_bp_s", "cholesterol",
    "fasting_blood_sugar", "resting_ecg", "max_heart_rate",
    "exercise_angina", "oldpeak", "st_slope", "target",
    "age_group", "chol_category"
]

# If some columns are missing, add them as NULL
for col in expected_cols:
    if col not in df.columns:
        df[col] = None

# Save to SQL (replace table if exists)
df.to_sql("patients", conn, if_exists="replace", index=False)

# ----------------------------
# Finish
# ----------------------------
conn.commit()
conn.close()
print("Database created and data loaded successfully!")
