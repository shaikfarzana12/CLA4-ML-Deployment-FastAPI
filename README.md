# ML Model Deployment Strategies and Model Serving using FastAPI (CLA-4)

Titanic survival classifier demonstrating **online**, **batch** and **streaming** inference, served through a **FastAPI** REST API.

**Dataset:** Titanic – Machine Learning from Disaster (Kaggle) – https://www.kaggle.com/competitions/titanic/data (`train.csv`, 891 rows; saved in `data/train.csv`).

## Project structure
| File | Purpose |
|---|---|
| `task1_train.py` | Preprocess, train RandomForest pipeline, report accuracy, save `model.pkl`, save held-out records |
| `task2_online.py` | Online inference – 1 unseen record |
| `task3_batch.py` | Batch inference – 20 unseen records → `batch_predictions.csv` |
| `task4_streaming.py` | Streaming inference – 10 unseen records, 1 s delay each |
| `app.py` | FastAPI app: `GET /health`, `POST /predict` (Pydantic validation) |
| `client.py` | Python client testing the API (valid + invalid request) |
| `report.pdf` | Observations report (Task 6) |
| `screenshots/`, `outputs/` | Evidence of every run |

## Execution steps
```bash
pip install -r requirements.txt
python task1_train.py        # trains once, creates model.pkl + data/unseen_records.csv
python task2_online.py
python task3_batch.py
python task4_streaming.py
uvicorn app:app --reload     # open http://127.0.0.1:8000/docs
python client.py             # in a second terminal
```

## Approach
- Features: Pclass, Sex, Embarked, Age, SibSp, Parch, Fare.
- Preprocessing (inside the pipeline): median imputation + scaling for numeric, most-frequent imputation + one-hot for categorical.
- Model: RandomForestClassifier. 80/20 stratified split (`random_state=42`).
- The 179 held-out test records (never used in training) are the "unseen" records for all inference demos.
- One saved pipeline (`model.pkl`) is reused by every inference strategy and by FastAPI.

## Results
- **Test accuracy: 79.89 %** (179 held-out passengers).
- Online: one record predicted (`screenshots/01_online_inference.png`).
- Batch: 20 records → `batch_predictions.csv` (85 % match with actual labels on this sample).
- Streaming: 10 records, one per second (`screenshots/03_streaming_inference.png`).
- API: `/health` → 200; valid `/predict` → 200; invalid → 422 with field-level errors (`screenshots/06–09`).

Note: Swagger screenshots show `localhost:8001` because they were captured on a second local port; run on port 8000 as above.
