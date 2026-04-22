# Price Engine — Vehicle Valuation Ensemble

> Predicts used vehicle market value from spec inputs. Dual-model ensemble (XGBoost + Neural Network) served via FastAPI, with an editorial-style static frontend and a chained MMR imputation model so the end user only enters what they actually know about their car.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688)
![XGBoost](https://img.shields.io/badge/XGBoost-ensemble-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-keras-FF6F00)

---

## What it does

Takes vehicle specifications — year, make, model, trim, body style, transmission, condition, odometer, color, interior, state — and returns a predicted market selling price in USD. Two ways to use it:

- **Single vehicle**, via a form on the frontend
- **Batch CSV upload**, returning the same file with a `predicted_price` column appended

A secondary XGBoost model estimates wholesale value (MMR) server-side, so the user never has to know what MMR is or look it up on an auction service. The main ensemble treats imputed MMR identically to measured MMR, and falls back to real MMR when it's present in an uploaded CSV.

## Quick demo

```bash
# terminal 1 — backend
uvicorn main:app --reload

# terminal 2 — frontend
cd frontend && python -m http.server 8080
```

Open `http://localhost:8080`.

---

## Architecture

```
         ┌──────────────┐
         │   Frontend   │    static HTML/CSS/JS, no build step
         │   (:8080)    │    vanilla JS, custom cursor, editorial layout
         └──────┬───────┘
                │ fetch JSON / multipart
                ▼
         ┌──────────────┐
         │   FastAPI    │    CORS-enabled, single-file app
         │   (:8000)    │
         └──────┬───────┘
                │
                ▼
    ┌───────────────────────┐
    │  Prediction Pipeline  │
    │                       │
    │  1. clean + validate  │
    │  2. MMR imputation ───│──> mmr_estimator.joblib (XGBoost)
    │  3. feature eng.      │
    │  4. preprocessor.transform
    │  5. ensemble predict ─│──> xgboost_model.joblib   (80% weight)
    │                       │──> neural_network.keras   (20% weight)
    │  6. weighted combine  │
    │  7. return USD        │
    └───────────────────────┘
```

### Why a chained model?

MMR (Manheim Market Report) is the single strongest numeric predictor in the dataset — but it's a wholesale auction value that requires a paid subscription to look up. Asking a user for it is hostile UX.

Dropping it would tank accuracy. Instead, a secondary XGBoost regressor estimates MMR from easier-to-know features (year, make, model, trim, body, transmission, condition, odometer). At inference time, missing MMR is filled in per-row before the main ensemble runs. If real MMR is present (from a CSV upload), it's used directly — no accuracy lost.

---

## Methodology

### Data

- **Source:** `car_prices.csv` — 558,837 auction records with make, model, trim, body, transmission, condition, odometer, color, interior, state, seller, MMR, and selling price.
- **Split:** 80/10/10 train/val/test. All preprocessing was fit on the training split only to avoid data leakage.

### Feature engineering

Six engineered features, on top of the raw columns:

| Feature | Motivation |
|---|---|
| `car_age` | Non-linear depreciation is easier to learn than raw `year` |
| `log_odometer` | Compresses the mileage range — the jump from 10k→20k matters more than 190k→200k |
| `miles_per_year` | Proxy for how hard the vehicle was used |
| `mmr_log` | Same reasoning — compresses the extreme price tiers |
| `mmr_per_mile` | Captures value retention per mile driven |
| `condition × mmr` | Interaction: good condition on a premium car commands a premium |

### Target transform

Selling price is transformed with `log1p` before training, and predictions are inverted with `expm1`. This stabilizes the loss across price tiers — a $500 error on a $5,000 car matters more than the same $500 error on a $50,000 car, and log-space training reflects that.

### Models

- **XGBoost** — tree-based, handles categorical structure and feature interactions cleanly.
- **Neural network** — Keras dense model with residual connections and embedding layers for high-cardinality categoricals (`model`, `trim`, `seller`). Picks up continuous patterns that trees miss.
- **Ensemble weights** — fit on the validation split, not chosen by hand.

### MMR estimator

A separate scikit-learn pipeline: ordinal-encoded categoricals fed to a 500-tree XGBoost regressor, trained on the same dataset with MMR as the target. Evaluated separately before being chained into the main pipeline.

---

## Results

Run `python train_mmr_estimator.py` to reproduce the MMR estimator and print its metrics. The main ensemble metrics are produced by `refined_Neural_Network.py`.

*(Fill in with your actual test-set numbers before publishing — placeholders below.)*

| Model | Test MAE | Test RMSE | Test R² |
|---|---|---|---|
| XGBoost (solo) | $XXX | $XXXX | 0.XX |
| Neural Network (solo) | $XXX | $XXXX | 0.XX |
| **Ensemble** | **$XXX** | **$XXXX** | **0.XX** |
| MMR estimator | $XXX | — | — |

---

## Project structure

```
├── main.py                       FastAPI app — loads all 5 models, serves 2 endpoints
├── refined_Neural_Network.py     Training script for the main ensemble
├── train_mmr_estimator.py        Training script for the MMR imputation model
│
├── neural_network_model.keras    trained NN (commit with Git LFS if > 100MB)
├── xgboost_model.joblib          trained XGBoost
├── nn_preprocess.joblib          sklearn ColumnTransformer
├── ensemble_weights.joblib       {"xgb": 0.8, "nn": 0.2}
├── mmr_estimator.joblib          secondary XGBoost for MMR imputation
│
└── frontend/
    ├── index.html                single-file site — HTML, CSS, JS inline
    └── assets/
        ├── fonts/                Silk Serif + Plain (woff2)
        └── img/                  hero + section imagery
```

## Tech stack

- **ML:** XGBoost, TensorFlow/Keras, scikit-learn (ColumnTransformer, OrdinalEncoder, Pipeline), pandas, NumPy, joblib
- **Backend:** FastAPI, Uvicorn, python-multipart
- **Frontend:** vanilla HTML/CSS/JS — no build step, no framework. Intersection Observer for scroll reveals, Fetch API for calls, custom CSS cursor, `woff2` display fonts

## Setup

```bash
# 1. clone
git clone https://github.com/<your-username>/price-engine.git
cd price-engine

# 2. install python deps
pip install -r requirements.txt

# 3. train the MMR estimator (one-time, ~1-2 min)
#    requires car_prices.csv in the project root
python train_mmr_estimator.py

# 4. (optional) retrain the main ensemble
python refined_Neural_Network.py

# 5. run the backend
uvicorn main:app --reload

# 6. serve the frontend (separate terminal)
cd frontend && python -m http.server 8080
```

## API

### `POST /predict-single`

Predicts one vehicle. JSON body, all fields optional except the essentials (year, make, model, condition, odometer).

```json
{
  "year": 2019,
  "make": "audi",
  "model": "a4",
  "trim": "premium plus",
  "body": "sedan",
  "transmission": "automatic",
  "condition": 35,
  "odometer": 45000,
  "color": "black",
  "interior": "black",
  "state": "pa"
}
```

Response:
```json
{
  "predicted_price": 24875.43,
  "model_weights": { "xgboost": 0.80, "neural_network": 0.20 }
}
```

### `POST /predict`

Upload a CSV (or XLSX). Returns the same file with an appended `predicted_price` column.

```bash
curl -F "file=@your_vehicles.csv" http://localhost:8000/predict -o predicted.csv
```

---

## Design

The frontend's aesthetic is editorial — inspired by studios like Obys Agency. Dark background (`#0e0e0e`), warm cream foreground (`#f4efe3`), a single sharp lime accent (`#c9ff3a`) used only for emphasis moments (active states, the final price reveal). Silk Serif italic for display type against Plain Regular for structural UI. Custom cursor with hover states, a counter preloader, scroll-triggered reveals, and a marquee ticker for rhythm breaks.

No framework. No build step. Everything is in one `index.html`.

## Roadmap

- [ ] Deploy backend to Render (currently local-only)
- [ ] Deploy frontend to Vercel
- [ ] Add prediction confidence intervals (quantile regression variant)
- [ ] Cache the MMR estimator output for common vehicle configurations
- [ ] Model monitoring — log prediction distributions to detect drift

## About

Built as the final project for the ML Web Applications course at Saint Joseph's University, Spring 2026.

**Author:** Kamil Traczyk — [LinkedIn](https://linkedin.com/in/kamil-traczyk-036ba7274)

## License

MIT — see `LICENSE` for details. The vehicle dataset is from Kaggle's [Used Car Auction Prices](https://www.kaggle.com/datasets/tunguz/used-car-auction-prices) and retains its original license.
