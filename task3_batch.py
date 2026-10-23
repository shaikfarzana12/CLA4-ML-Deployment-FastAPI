"""Task 3: Batch Inference - 20 unseen records predicted together."""
import joblib
import pandas as pd

model = joblib.load("model.pkl")
unseen = pd.read_csv("data/unseen_records.csv")
batch = unseen.iloc[1:21].copy()               # 20 unseen records
features = batch.drop(columns=["Survived_actual"])

batch["Predicted"] = model.predict(features)   # all 20 in one call
batch["Survival_Probability"] = model.predict_proba(features)[:, 1].round(3)
batch.to_csv("batch_predictions.csv", index=False)

print("=== BATCH INFERENCE (20 records) ===")
pd.set_option("display.width", 200)
print(batch.to_string())
print(f"\nSaved {len(batch)} predictions -> batch_predictions.csv")
print(f"Batch accuracy vs actual: {(batch['Predicted'] == batch['Survived_actual']).mean():.2%}")
