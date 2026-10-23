"""Task 4: Streaming Inference - 10 unseen records, one at a time, 1s apart."""
import time
import joblib
import pandas as pd

model = joblib.load("model.pkl")
unseen = pd.read_csv("data/unseen_records.csv")
stream = unseen.iloc[21:31]                    # 10 unseen records

print("=== STREAMING INFERENCE (1 record / second) ===")
for i, (_, row) in enumerate(stream.iterrows(), start=1):
    time.sleep(1)                              # simulate arrival delay
    features = row.drop("Survived_actual").to_frame().T
    pred = int(model.predict(features)[0])
    prob = float(model.predict_proba(features)[0][1])
    print(f"[Record {i:02d}] Pclass={row['Pclass']} Sex={row['Sex']} Age={row['Age']} "
          f"Fare={row['Fare']:.2f} Embarked={row['Embarked']} -> "
          f"Prediction: {'Survived' if pred else 'Did not survive'} "
          f"(p={prob:.3f}) | Actual: {int(row['Survived_actual'])}", flush=True)
print("Stream finished.")
