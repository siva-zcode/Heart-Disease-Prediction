import os

print("STEP 1: Cleaning data...")
os.system("python scripts/data_cleaning.py")

print("STEP 2: Loading database...")
os.system("python scripts/load_sqlite.py")

print("STEP 3: Running EDA...")
os.system("python scripts/eda.py")

print("STEP 4: Training ML model...")
os.system("python scripts/ml_prediction.py")

print("STEP 5: Launching dashboard...")
os.system("streamlit run sql_dashboard.py")
