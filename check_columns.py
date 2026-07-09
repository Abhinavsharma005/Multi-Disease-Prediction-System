import joblib

def print_details():
    print("--- Symptom Features ---")
    sym_features = joblib.load("models/symptom_features.joblib")
    print(f"Count: {len(sym_features)}")
    print(sym_features[:10], "... etc.")

    print("\n--- Heart Columns ---")
    heart_cols = joblib.load("models/heart_columns.joblib")
    print(f"Count: {len(heart_cols)}")
    print(heart_cols)

    print("\n--- Diabetes Columns ---")
    diab_cols = joblib.load("models/diabetes_columns.joblib")
    print(f"Count: {len(diab_cols)}")
    print(diab_cols)

    print("\n--- Mental Health Columns ---")
    mental_cols = joblib.load("models/mental_columns.joblib")
    print(f"Count: {len(mental_cols)}")
    print(mental_cols)
    
    print("\n--- Mental Health Categorical Classes ---")
    encoders = joblib.load("models/mental_encoders.joblib")
    for col, enc in encoders.items():
        print(f"{col}: {list(enc.classes_)}")

if __name__ == '__main__':
    print_details()
