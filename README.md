# MP152 — Cinema Show Occupancy (Patna)

## Group Information

| Field | Value |
|---|---|
| **Group ID** | G215 |
| **Problem ID** | MP152 |
| **Problem Title** | Cinema Show Occupancy (Patna) |
| **Course** | SCSE3040 MLOps |

## Members

| Name | Student ID |
|---|---|
| YashVardhan Patel | S24CSEU0987 |
| Rishita Sinha | S24CSEU0992 |
| Atharv Krishna | S24CSEU0989 |
| Rajan Bharadwaj | S24CSEU0973 |

## Problem Description

Predict the occupancy percentage (`occupancy_pct`) of cinema shows in Patna given features like ticket price, days since release, star rating, show hour, screen seats, distance to metro, language, and location. This is a **regression** problem evaluated with **MAE** (Mean Absolute Error).

## Repository Structure

```
G215_MP152/
├── README.md
├── requirements.txt
├── params.yaml
├── findings.yaml
├── Dockerfile
├── .dockerignore
├── .dvcignore
├── dvc.yaml
├── mlflow.db               # committed MLflow tracking DB
├── data/
│   ├── raw/                # DVC-tracked raw data
│   │   ├── train.csv
│   │   ├── later.csv
│   │   └── holdout_features.csv
├── src/
│   └── mp152_cinema/
│       ├── __init__.py
│       ├── cleaning.py     # Data cleaning & audit
│       ├── training.py     # Model training & MLflow logging
│       ├── evaluation.py   # Metrics: MAE, PSI
│       └── predict.py      # Load champion & run predictions
├── app/
│   └── main.py             # FastAPI service
├── tests/
│   ├── test_cleaning.py
│   ├── test_api.py
│   └── test_predictions.py
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── reports/
│   └── findings.yaml
├── predictions/
│   └── holdout_predictions.csv
└── .github/
    └── workflows/
        └── ci.yml
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run data audit
python src/mp152_cinema/cleaning.py

# Train models (logs to MLflow)
python src/mp152_cinema/training.py

# Run monitoring & retraining
python src/mp152_cinema/evaluation.py

# Generate holdout predictions
python src/mp152_cinema/predict.py

# Run tests
pytest tests/ -v

# Start API locally
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Build Docker image
docker build -t mp152-api:1.0 .
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check → `{"status": "ok"}` |
| `POST` | `/predict` | Predict occupancy_pct |

### POST /predict

Request body (JSON):
```json
{
  "ticket_price_rs": 350,
  "days_since_release": 14,
  "star_rating": 3.5,
  "show_hour": 18,
  "screen_seats": 250,
  "distance_to_metro_km": 2.0,
  "language": "hindi",
  "location": "Mall"
}
```

Validation rules (returns 422 on violation):
- `distance_to_metro_km` ≤ 8.5
- `ticket_price_rs` ≥ 50
