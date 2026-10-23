"""Python client to test the FastAPI prediction endpoint (start the server first)."""
import requests

BASE = "http://127.0.0.1:8000"

print("GET /health ->", requests.get(f"{BASE}/health").json())

valid = {"Pclass": 1, "Sex": "female", "Age": 29, "SibSp": 0,
         "Parch": 0, "Fare": 80.5, "Embarked": "C"}
r = requests.post(f"{BASE}/predict", json=valid)
print("\nValid request  :", valid)
print("Status", r.status_code, "->", r.json())

invalid = {"Pclass": 5, "Sex": "alien", "Age": -4, "SibSp": 0,
           "Parch": 0, "Fare": "abc", "Embarked": "X"}
r = requests.post(f"{BASE}/predict", json=invalid)
print("\nInvalid request:", invalid)
print("Status", r.status_code, "->", r.json())
