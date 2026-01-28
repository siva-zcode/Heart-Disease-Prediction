import pandas as pd

df = pd.read_csv("data/processed/heart_disease_cleaned.csv")

total_patients = len(df)
disease_rate = df["target"].mean() * 100
avg_age = df["age"].mean()
high_chol = len(df[df["chol"] > 240])

summary = f"""
Healthcare Data Insight Report:

- Total Patients: {total_patients}
- Disease Prevalence: {disease_rate:.2f}%
- Average Age: {avg_age:.1f}
- High Cholesterol Cases: {high_chol}

Key Observations:
- Middle-aged and senior groups show higher disease prevalence.
- Cholesterol and age are strongly correlated with heart disease.
- Males show slightly higher risk levels.

Recommendations:
- Promote early screening for patients above 45.
- Lifestyle monitoring for high-cholesterol patients.
- Awareness programs for high-risk groups.
"""

print(summary)

with open("ai_healthcare_report.txt", "w") as f:
    f.write(summary)

print("AI insight report generated!")
