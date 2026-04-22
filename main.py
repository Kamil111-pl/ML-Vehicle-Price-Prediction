#FastAPI Backend for Vehicle Price Prediction
#This loads the trained ensemble model (Neural Network + XGBoost)
#and serves predictions through two endpoints:
#1. /predict - upload a CSV file and get predictions for all cars
#2. /predict-single - enter one car's info and get a price
#
#NEW: mmr and seller are now auto-filled server-side so the frontend
#doesn't need to ask the user for them.
# - mmr  → estimated by a secondary xgboost regressor (mmr_estimator.joblib)
# - seller → defaulted to "unknown"

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import io

app = FastAPI()

#allow the frontend to connect from any common local dev origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "null",  # file:// origin
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# STEP 1 — LOAD ALL MODELS (happens once at startup)

print("Loading models...")
nn_model      = tf.keras.models.load_model("neural_network_model.keras")
xgb_model     = joblib.load("xgboost_model.joblib")
preprocessor  = joblib.load("nn_preprocess.joblib")
weights       = joblib.load("ensemble_weights.joblib")
mmr_estimator = joblib.load("mmr_estimator.joblib")   # NEW

xgb_w = weights["xgb"]
nn_w  = weights["nn"]
print(f"Models loaded. Ensemble weights: {xgb_w:.0%} XGBoost + {nn_w:.0%} NN")
print("MMR estimator loaded (secondary xgboost).")

CURRENT_YEAR = 2026


# STEP 2 — CONDITION FIX HELPER
# the condition column mixes two scales: 1-5 (star rating) and 11-49 (manheim)
# both normalized to 0-1 so the model sees one consistent scale
def fix_condition_value(v):
    if pd.isna(v):
        return np.nan
    if v <= 5:
        return v / 5.0
    return (v - 10.0) / 39.0


# STEP 3 — MMR ESTIMATION (NEW)
# if the user didn't give us mmr, we estimate it from the easier features.
# this keeps the main ensemble pipeline intact — it still sees an mmr column,
# just a predicted one instead of a measured one.
def estimate_mmr(raw_df):
    # work on a copy so we don't mutate the caller's dataframe
    df = raw_df.copy()

    # fix condition the same way the estimator was trained
    df["condition"] = pd.to_numeric(df["condition"], errors="coerce").apply(fix_condition_value)

    # make sure numerics are actually numbers
    for col in ["year", "odometer"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # make sure all expected columns exist (fill missing with NaN)
    needed = ["year", "odometer", "condition", "make", "model", "trim", "body", "transmission"]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    # predict mmr for each row
    return mmr_estimator.predict(df[needed])


# STEP 4 — FEATURE ENGINEERING
# same features the main ensemble was trained on

def add_features(df):

    # normalize condition to 0-1 for the main ensemble
    if "condition" in df.columns:
        df["condition"] = pd.to_numeric(df["condition"], errors="coerce").apply(fix_condition_value)

    # make sure numerics are numbers
    for col in ["year", "odometer", "mmr"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["car_age"]             = (CURRENT_YEAR - df["year"]).clip(lower=0)
    df["log_odometer"]        = np.log1p(df["odometer"])
    df["miles_per_year"]      = df["odometer"] / (df["car_age"] + 1)
    df["mmr_log"]             = np.log1p(df["mmr"])
    df["mmr_per_mile"]        = df["mmr"] / (df["odometer"] + 1)
    df["condition_times_mmr"] = df["condition"] * df["mmr"]

    return df


# STEP 5 — MAIN PREDICT PIPELINE

def predict_prices(df):

    # clean column names
    df.columns = df.columns.str.strip().str.lower()

    # drop columns the model doesn't need
    for col in ["vin", "saledate", "sellingprice"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # fix string columns
    if "body" in df.columns:
        df["body"] = df["body"].astype(str).str.strip().str.lower()
    if "transmission" in df.columns:
        df["transmission"] = df["transmission"].astype(str).str.strip().str.lower()
        df["transmission"] = df["transmission"].where(
            df["transmission"].isin(["automatic", "manual"]), other=np.nan
        )

    # ─── AUTO-FILL MISSING MMR ───────────────────────────────
    # if mmr is completely missing OR all NaN, estimate it
    if "mmr" not in df.columns or df["mmr"].isna().all():
        print(f"  mmr missing — estimating for {len(df)} row(s)")
        df["mmr"] = estimate_mmr(df)
    else:
        # some rows have mmr, some don't — estimate the missing ones
        mask = pd.to_numeric(df["mmr"], errors="coerce").isna()
        if mask.any():
            print(f"  estimating mmr for {mask.sum()} row(s) with missing values")
            df.loc[mask, "mmr"] = estimate_mmr(df.loc[mask])

    # ─── AUTO-FILL MISSING SELLER ────────────────────────────
    if "seller" not in df.columns:
        df["seller"] = "unknown"
    else:
        df["seller"] = df["seller"].fillna("unknown").astype(str)

    # add engineered features
    df = add_features(df)

    # apply preprocessing (same pipeline from training)
    processed = preprocessor.transform(df)
    if hasattr(processed, "toarray"):
        processed = processed.toarray().astype(np.float32)

    # predict with both models
    nn_preds  = np.expm1(nn_model.predict(processed, verbose=0).flatten())
    xgb_preds = np.expm1(xgb_model.predict(processed))

    # weighted ensemble
    ensemble_preds = xgb_w * xgb_preds + nn_w * nn_preds

    return ensemble_preds


# ─────────────────────────────────────────────────────────────
# ENDPOINT 1: Upload CSV and get predictions back
# ─────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict_from_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return {"error": "Please upload a CSV or Excel file"}

        original_df = df.copy()

        predictions = predict_prices(df)

        original_df.columns = original_df.columns.str.strip().str.lower()
        original_df["predicted_price"] = predictions.round(2)

        output = io.BytesIO()
        original_df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=predicted_prices_{file.filename}"}
        )

    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────
# ENDPOINT 2: Single car manual input (JSON)
# ─────────────────────────────────────────────────────────────
@app.post("/predict-single")
async def predict_single_car(car: dict):
    try:
        df = pd.DataFrame([car])
        predictions = predict_prices(df)

        return {
            "predicted_price": round(float(predictions[0]), 2),
            "model_weights": {"xgboost": xgb_w, "neural_network": nn_w}
        }

    except Exception as e:
        return {"error": str(e)}


# ─── RUN SERVER ──────────────────────────────────────────────
# to start: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)