import pandas as pd

df = pd.read_csv("mental_health_risk_dataset.csv")
print(df.dtypes)
print("\nUnique values for columns not in label encoders list:")
non_encoder_cols = [c for c in df.columns if c not in ['gender', 'marital_status', 'education_level', 'employment_status', 'panic_attack_history', 'family_history_mental_illness', 'previous_mental_health_diagnosis', 'therapy_history', 'substance_use', 'mental_health_risk']]
for col in non_encoder_cols:
    print(f"{col}: {df[col].unique()[:10]} (dtype: {df[col].dtype})")
