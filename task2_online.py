"""Task 2: Online Inference - one unseen passenger, one prediction."""
import joblib
import pandas as pd

model = joblib.load("model.pkl")
unseen = pd.read_csv("data/unseen_records.csv")
record = unseen.iloc[[0]]                      # one unseen record
features = record.drop(columns=["Survived_actual"])

pred = int(model.predict(features)[0])
prob = float(model.predict_proba(features)[0][1])

print("=== ONLINE INFERENCE (single record) ===")
print("Passenger details:")
print(features.iloc[0].to_string())
print(f"\nPredicted: {pred} ({'Survived' if pred else 'Did not survive'})")
print(f"Survival probability: {prob:.3f}")
print(f"Actual   : {int(record['Survived_actual'].iloc[0])}")
