"""Task 5: FastAPI model serving. Run: uvicorn app:app --reload"""
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Titanic Survival API", version="1.0")
model = joblib.load("model.pkl")               # same pipeline from Task 1


class Passenger(BaseModel):
    Pclass: Literal[1, 2, 3] = Field(..., description="Ticket class (1, 2 or 3)")
    Sex: Literal["male", "female"]
    Age: Optional[float] = Field(None, ge=0, le=120)
    SibSp: int = Field(..., ge=0, le=10)
    Parch: int = Field(..., ge=0, le=10)
    Fare: float = Field(..., ge=0)
    Embarked: Literal["S", "C", "Q"]

    model_config = {"json_schema_extra": {"example": {
        "Pclass": 1, "Sex": "female", "Age": 29, "SibSp": 0,
        "Parch": 0, "Fare": 80.5, "Embarked": "C"}}}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(p: Passenger):
    df = pd.DataFrame([p.model_dump()])
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])
    return {"prediction": pred,
            "result": "Survived" if pred else "Did not survive",
            "survival_probability": round(prob, 4)}
