"""Task 1: Model Training - Titanic survival classifier.
Loads data/train.csv, preprocesses, trains once, evaluates (accuracy),
saves the full pipeline to model.pkl and the held-out (unseen) records
to data/unseen_records.csv for the inference tasks."""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["Age", "SibSp", "Parch", "Fare"]
CATEGORICAL = ["Pclass", "Sex", "Embarked"]
FEATURES = CATEGORICAL + NUMERIC

df = pd.read_csv("data/train.csv")
print("Dataset shape:", df.shape)
print("Missing values:\n", df[FEATURES + ["Survived"]].isna().sum(), sep="")

X, y = df[FEATURES], df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing: impute + scale numbers, impute + one-hot categories
preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), NUMERIC),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
])
pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(n_estimators=200, max_depth=6,
                                     random_state=42)),
])
pipeline.fit(X_train, y_train)

acc = accuracy_score(y_test, pipeline.predict(X_test))
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Test accuracy: {acc:.4f}")

joblib.dump(pipeline, "model.pkl")
print("Saved pipeline -> model.pkl")

# Held-out records (never seen in training) used for all inference demos
unseen = X_test.copy()
unseen["Survived_actual"] = y_test
unseen.to_csv("data/unseen_records.csv", index=False)
print("Saved", len(unseen), "unseen records -> data/unseen_records.csv")
