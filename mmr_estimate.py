# train_mmr_estimator.py
# ─────────────────────────────────────────────────────────────
# ONE-TIME SCRIPT — run this once to train a small XGBoost model
# that estimates MMR (wholesale market value) from the easy-to-know
# vehicle specs. The main.py backend loads this to auto-fill MMR
# when the user submits a single-vehicle prediction.
#
# Run:  python train_mmr_estimator.py
# Out:  mmr_estimator.joblib  (saved next to the other model files)
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


# STEP 1 — CONFIG
# change this if your training CSV has a different name / path
DATA_PATH = "car_prices.csv"
OUT_PATH  = "mmr_estimator.joblib"


# STEP 2 — LOAD DATA
print("loading data...")
df = pd.read_csv(DATA_PATH, on_bad_lines="skip")
df.columns = df.columns.str.strip().str.lower()
print(f"  loaded {len(df):,} rows")


# STEP 3 — CLEAN
# the same cleaning logic from main.py, so the estimator sees
# values in the same shape it'll see at inference time
if "body" in df.columns:
    df["body"] = df["body"].astype(str).str.strip().str.lower()
if "transmission" in df.columns:
    df["transmission"] = df["transmission"].astype(str).str.strip().str.lower()
    df["transmission"] = df["transmission"].where(
        df["transmission"].isin(["automatic", "manual"]), other=np.nan
    )

# same condition normalization as the main model
def fix_condition(v):
    if pd.isna(v):  return np.nan
    if v <= 5:      return v / 5.0
    return (v - 10.0) / 39.0

df["condition"] = pd.to_numeric(df["condition"], errors="coerce").apply(fix_condition)

# make sure numerics are numeric
for col in ["year", "odometer", "mmr"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# STEP 4 — PICK FEATURES
# everything we can realistically ask a user for
NUM_COLS = ["year", "odometer", "condition"]
CAT_COLS = ["make", "model", "trim", "body", "transmission"]
TARGET   = "mmr"

# drop rows missing the essentials
df = df.dropna(subset=NUM_COLS + CAT_COLS + [TARGET])
print(f"  {len(df):,} rows after dropna")

X = df[NUM_COLS + CAT_COLS]
y = df[TARGET].values


# STEP 5 — PIPELINE
# categoricals → ordinal encoding (xgboost handles it fine and it's fast)
# numerics → passthrough
# unknown_value=-1 means new/unseen categories at inference won't crash
pre = ColumnTransformer([
    ("num", "passthrough", NUM_COLS),
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
           CAT_COLS),
])

xgb = XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)

pipe = Pipeline([("pre", pre), ("xgb", xgb)])


# STEP 6 — TRAIN
print("training mmr estimator...")
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1, random_state=42)
pipe.fit(X_tr, y_tr)

# STEP 7 — EVALUATE
preds = pipe.predict(X_te)
mae   = np.mean(np.abs(preds - y_te))
mape  = np.mean(np.abs(preds - y_te) / np.maximum(y_te, 1)) * 100
print(f"  mmr estimator MAE : ${mae:,.0f}")
print(f"  mmr estimator MAPE: {mape:.1f}%")


# STEP 8 — SAVE
joblib.dump(pipe, OUT_PATH)
print(f"saved → {OUT_PATH}")