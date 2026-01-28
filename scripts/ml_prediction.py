import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from pathlib import Path
import numpy as np
import os

# -----------------------------
# Load cleaned dataset
# -----------------------------
df = pd.read_csv("data/processed/heart_disease_cleaned.csv")

# -----------------------------
# Prepare features and target
# -----------------------------
y = df["target"]

# Drop target & age_group (non-numeric)
X = df.drop(columns=["target", "age_group"])

# One-hot encode categorical columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
if len(categorical_cols) > 0:
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# -----------------------------
# Scale features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Split dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Train Random Forest
# -----------------------------
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_cv = cross_val_score(rf_model, X_scaled, y, cv=5).mean()
print(f"Random Forest Accuracy: {rf_acc*100:.2f}% | 5-Fold CV: {rf_cv*100:.2f}%")

# -----------------------------
# Train XGBoost
# -----------------------------
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_cv = cross_val_score(xgb_model, X_scaled, y, cv=5).mean()
print(f"XGBoost Accuracy: {xgb_acc*100:.2f}% | 5-Fold CV: {xgb_cv*100:.2f}%")

# -----------------------------
# Choose best model
# -----------------------------
best_model = rf_model if rf_acc >= xgb_acc else xgb_model
best_name = 'Random Forest' if best_model == rf_model else 'XGBoost'
print(f"Best Model: {best_name}")

# -----------------------------
# Confusion matrix
# -----------------------------
cm = confusion_matrix(y_test, best_model.predict(X_test))
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"{best_name} Confusion Matrix")
plt.tight_layout()
Path("images").mkdir(exist_ok=True)
plt.savefig("images/confusion_matrix.png")
plt.close()

# -----------------------------
# Classification Report
# -----------------------------
class_report = classification_report(y_test, best_model.predict(X_test), output_dict=True)
print("\nClassification Report:\n", classification_report(y_test, best_model.predict(X_test)))

# -----------------------------
# ROC Curve and AUC
# -----------------------------
y_prob = best_model.predict_proba(X_test)[:,1]  # probability of positive class
auc_score = roc_auc_score(y_test, y_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0,1],[0,1],'k--')  # random line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"{best_name} ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("images/roc_curve.png")
plt.close()
print(f"\nROC AUC Score: {auc_score:.2f}")

# -----------------------------
# Feature Importance
# -----------------------------
feat_imp_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(x="Importance", y="Feature", data=feat_imp_df, palette="viridis")
plt.title(f"{best_name} Feature Importance")
plt.tight_layout()
plt.savefig("images/feature_importance.png")
plt.close()

feat_imp_df.to_csv("feature_importance.csv", index=False)

# -----------------------------
# Save model, scaler, feature columns
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/final_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
with open("models/feature_columns.json", "w") as f:
    json.dump(X.columns.tolist(), f)

# -----------------------------
# Save evaluation metrics
# -----------------------------
eval_metrics = {
    "Model": best_name,
    "Accuracy": round(accuracy_score(y_test, best_model.predict(X_test)), 4),
    "5Fold_CV_Accuracy": round(cross_val_score(best_model, X_scaled, y, cv=5).mean(), 4),
    "ROC_AUC": round(auc_score, 4),
    "Classification_Report": class_report
}

with open("models/evaluation_metrics.json", "w") as f:
    json.dump(eval_metrics, f, indent=4)

print("\nEvaluation metrics saved to 'models/evaluation_metrics.json'")

# -----------------------------
# Save predictions with risk categories
# -----------------------------
pred_prob = best_model.predict_proba(X_scaled)[:,1]
pred_df = pd.DataFrame(X, columns=X.columns)
pred_df["Actual"] = y.values
pred_df["Predicted"] = best_model.predict(X_scaled)
pred_df["Risk_Percentage"] = (pred_prob*100).round(2)

def risk_category(prob):
    if prob < 30:
        return "Low"
    elif prob < 60:
        return "Medium"
    else:
        return "High"

pred_df["Risk_Category"] = pred_df["Risk_Percentage"].apply(risk_category)
pred_df.to_csv("predicted_disease_risk.csv", index=False)

# -----------------------------
# Save top 3 features affecting disease risk
# -----------------------------
top_features = feat_imp_df.head(3)["Feature"].tolist()
with open("top_features.txt", "w") as f:
    f.write(",".join(top_features))
print(f"Top features affecting disease risk: {', '.join(top_features)}")
print("ML predictions, risk categories, feature importance, and evaluation metrics saved successfully.")
