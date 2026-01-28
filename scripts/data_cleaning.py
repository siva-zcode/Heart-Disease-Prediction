import pandas as pd
import numpy as np

# -----------------------------
# Load raw data
# -----------------------------
df = pd.read_csv("data/raw/heart_disease_raw.csv")
print("Initial shape:", df.shape)

# -----------------------------
# Standardize column names
# -----------------------------
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# -----------------------------
# Remove duplicate records
# -----------------------------
dup_count = df.duplicated().sum()
df = df.drop_duplicates()
print(f"Duplicates removed: {dup_count}")

# -----------------------------
# Fix data types
# -----------------------------
numeric_cols = [
    "age", "resting_bp_s", "cholesterol", "max_heart_rate",
    "oldpeak"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -----------------------------
# Handle missing values
# -----------------------------
missing_before = df.isnull().sum()

# Fill numeric columns with median
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with mode
categorical_cols = ["sex", "chest_pain_type", "fasting_blood_sugar",
                    "resting_ecg", "exercise_angina", "st_slope", "target"]
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

missing_after = df.isnull().sum()
print("\nMissing values before cleaning:\n", missing_before)
print("\nMissing values after cleaning:\n", missing_after)

# -----------------------------
# Remove impossible values
# -----------------------------
df = df[(df["age"] >= 0) & (df["age"] <= 120)]
df = df[df["cholesterol"] > 0]
df = df[df["resting_bp_s"] > 0]
df = df[df["oldpeak"] >= 0]

print("Shape after removing impossible values:", df.shape)

# -----------------------------
# Handle outliers using IQR
# -----------------------------
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

for col in ["cholesterol", "resting_bp_s", "max_heart_rate", "oldpeak"]:
    before = df.shape[0]
    df = remove_outliers_iqr(df, col)
    after = df.shape[0]
    print(f"Outliers removed from {col}: {before - after}")

# -----------------------------
# Feature Engineering
# -----------------------------
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 30, 45, 60, 100],
    labels=["Young", "Adult", "Middle-Aged", "Senior"]
)

df["chol_category"] = pd.cut(
    df["cholesterol"],
    bins=[0, 200, 240, 1000],
    labels=["Normal", "Borderline", "High"]
)

# -----------------------------
# Final sanity checks
# -----------------------------
print("\nFinal shape:", df.shape)
print("\nFinal missing values:\n", df.isnull().sum())
print("\nFinal data types:\n", df.dtypes)

# -----------------------------
# Save cleaned data
# -----------------------------
df.to_csv("data/processed/heart_disease_cleaned.csv", index=False)
print("\nCleaned data saved to data/processed/heart_disease_cleaned.csv")
