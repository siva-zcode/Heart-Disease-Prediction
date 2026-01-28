import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load cleaned data
df = pd.read_csv("data/processed/heart_disease_cleaned.csv")

# Ensure images folder exists
images_path = Path("images")
images_path.mkdir(exist_ok=True)

# 1️⃣ Age vs Disease
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='age_group', hue='target')
plt.title("Age vs Disease")
plt.savefig(images_path / "age_vs_disease.png")
plt.close()

# 2️⃣ Cholesterol vs Disease
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='cholesterol', hue='target', bins=20, kde=True)
plt.title("Cholesterol vs Disease")
plt.savefig(images_path / "chol_vs_disease.png")
plt.close()

# 3️⃣ Correlation Heatmap (numeric only)
numeric_cols = df.select_dtypes(include='number')
plt.figure(figsize=(10,8))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig(images_path / "correlation_heatmap.png")
plt.close()

# 4️⃣ Disease Distribution
plt.figure(figsize=(6,4))
df['target'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title("Disease Distribution")
plt.savefig(images_path / "disease_distribution.png")
plt.close()

print("EDA images generated in images/ folder.")
