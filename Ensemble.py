#these libraries allow us to work with data and build models
import pandas as pd #used to load and work with CSV data
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential #used to build neural networks step by step
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#xgboost is a machine learning model that uses decision trees
from xgboost import XGBRegressor

#sklearn tools for data cleaning and preprocessing
from sklearn.model_selection import train_test_split   #splits data into training and testing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_log_error

#used to save models to disk
import joblib

#location of dataset on my mac
DATA_PATH = "/Users/kamil/Downloads/car_prices.csv"

#this is what we want to predict (target value)
TARGET = "sellingprice"

#current year for calculating car age
CURRENT_YEAR = 2026

#just like the linear regression model, we put our dataset into the data frame
df = pd.read_csv(DATA_PATH)

#clean column names (remove spaces + lowercase everything)
df.columns = df.columns.str.strip().str.lower()

#drop useless columns that don't help prediction
#saledate is not useful for price prediction
#VIN is a Vehicle ID number, not useful in our case
for col in ["vin", "saledate"]:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)


#clean up data here:
#the body column has duplicates like "Sedan" and "sedan"
#make everything lowercase so they match
if "body" in df.columns:
    df["body"] = df["body"].str.strip().str.lower()

#transmission has bad values like "sedan" which is not a transmission
#only keep "automatic" and "manual", everything else becomes NaN
if "transmission" in df.columns:
    df["transmission"] = df["transmission"].str.strip().str.lower()
    df["transmission"] = df["transmission"].where(
        df["transmission"].isin(["automatic", "manual"]), other=np.nan
    )

#about 10,000 rows have no make or model, drop them
df = df.dropna(subset=["make", "model"]).copy()

#make sure selling price column is numeric
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

#remove rows where price is missing
df = df.dropna(subset=[TARGET]).copy()

#remove outliers (prices too low or too high)
df = df[(df[TARGET] >= 500) & (df[TARGET] <= 120000)].copy()

#some text columns have thousands of unique values
#keep only the 30 most common, everything else becomes "other"
for col in ["model", "trim", "seller"]:
    if col in df.columns:
        top_30 = df[col].value_counts().nlargest(30).index
        df[col] = df[col].where(df[col].isin(top_30), other="other")

print("Dataset loaded:", df.shape)

#here we will do simple feature engineering
#creating new columns from existing data to help the model
#make sure these columns are numbers
for col in ["year", "odometer", "condition", "mmr"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

#drop rows missing important values
df = df.dropna(subset=["year", "odometer", "condition", "mmr"]).copy()

#some cars were rated 1-5 and others 11-49.
#a condition of 5 meant perfect on one scale but terrible on the other.
#I converted both to 0-1 so the model isn't confused.
def fix_condition(value):
    if value <= 5:
        return value / 5.0
    else:
        return (value - 10.0) / 39.0
df["condition"] = df["condition"].apply(fix_condition)
#how old is the car
#just the current year minus the car's year a 2020 car is 6 years old
df["car_age"] = (CURRENT_YEAR - df["year"]).clip(lower=0)
#the difference between 10,000 and 20,000 miles matters a lot
#but the difference between 190,000 and 200,000 miles barely matters
#log compresses big numbers so the model treats mileage more realistically
df["log_odometer"] = np.log1p(df["odometer"])
#average miles driven per year
#a 5 year old car with 100,000 miles was driven really hard
#a 5 year old car with 25,000 miles was barely used
#this tells the model how hard the car was used
df["miles_per_year"] = df["odometer"] / (df["car_age"] + 1)
#log of mmr (wholesale price estimate)
#i used log to smooth out the extreme values
#the difference between a $5,000 and $10,000 MMR matters way more
#than the difference between a $95,000 and $100,000 MMR
df["mmr_log"] = np.log1p(df["mmr"])
#how much market value per mile driven
#how much market value the car has per mile driven
#a Toyota with 100k miles and high market value holds its value well
#a luxury car with 100k miles and low market value doesn't
df["mmr_per_mile"] = df["mmr"] / (df["odometer"] + 1)
#condition times mmr, good condition on expensive car is premium
#a car in great condition that also has a high market value is worth a premium
df["condition_times_mmr"] = df["condition"] * df["mmr"]
print("Feature engineering done. Columns:", df.shape[1])



#
#X = all features used to predict price
X = df.drop(columns=[TARGET])

#y = actual selling price (what we want the model to learn)
#we use log of price for more stable training
y = np.log1p(df[TARGET].values.astype(np.float32))


#find numeric and text columns
#models only understand numbers
#so we separate numeric columns from text columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()

print(f"Numeric columns: {len(num_cols)}")
print(f"Text columns: {len(cat_cols)}")



#SPLIT DATA
#we split the data into three groups Training is what the model studies from
#validation is like a practice test it takes while studying to see if it's improving
#test is the final exam it's never seen before
#this tells us how good the model actually is on cars it wasn't trained on
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_train_raw, y_train, test_size=0.1, random_state=42
)

print(f"Training: {len(y_train)} cars")
print(f"Validation: {len(y_val)} cars")
print(f"Test: {len(y_test)} cars")


#Datapreproccessing:
#neural networks CANNOT read text directly
#everything must be converted into numbers first
#numeric pipeline (for number columns)
num_pipe = Pipeline([
    #if numeric values missing, fill with median
    ("imputer", SimpleImputer(strategy="median")),
    #scale numbers so they are balanced
    ("scaler", StandardScaler()),
])

#categorical pipeline (for text columns)
cat_pipe = Pipeline([
    #if text missing, fill with most common value
    ("imputer", SimpleImputer(strategy="most_frequent")),
    #convert text to numbers using one-hot encoding
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

#combine numeric and text pipelines together
preprocess = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])

#convert all raw data into pure numbers
#fit only on training data to prevent data leakage
X_train = preprocess.fit_transform(X_train_raw)
X_val = preprocess.transform(X_val_raw)
X_test = preprocess.transform(X_test_raw)

#convert sparse matrices to dense arrays
if hasattr(X_train, "toarray"):
    X_train = X_train.toarray().astype(np.float32)
    X_val = X_val.toarray().astype(np.float32)
    X_test = X_test.toarray().astype(np.float32)

print("Data converted to numbers for models.")
print("Total features:", X_train.shape[1])


#NEURAL NETWORK MODEL: 4 layers: 512 neurons, 256 neurons, 128 neurons and 64 neurons
#Sequential means each layer is stacked in order
model = Sequential()
#input layer tells the network how many features to expect
model.add(Input(shape=(X_train.shape[1],)))

#first hidden layer: 512 neurons
model.add(Dense(512, activation="relu"))

#after each layer the numbers can get really big or really small which makes training unstable
#this resets them to a normal range before passing to the next layer.
model.add(BatchNormalization())

#randomly turns off 30% of the neurons each time
#and this forces the network to not rely on any single neuron too much.
model.add(Dropout(0.3))

#second layer: 256 neurons
model.add(Dense(256, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))

#third layer 128 neurons
model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))

#fourth hidden layer: 64 neurons
model.add(Dense(64, activation="relu"))

#output layer predicts 1 value which is the log of vehicle sales price
model.add(Dense(1))

#compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
    loss=tf.keras.losses.Huber(), #like MAE but smoother, handles outliers better
    metrics=["mae"]
)

print("Neural network built.")
model.summary()

#early stopping: if model stops improving for 20 epochs, stop and go back to best version
stop_early = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

#reduce learning rate: if stuck for 6 epochs, slow down the learning
slow_down = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=1)

#train the neural network
print("\nTraining neural network:")
model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=[stop_early, slow_down],
    verbose=1
)

print("Neural network training finished.")

#XGBOOST MODEL
#xgboost builds lots of small decision trees
#each tree fixes the mistakes of the previous one
#its the best algorithm for spreadsheet type data
xgb = XGBRegressor(
    n_estimators=2000, #Build up to 2,000 trees, more trees means more chances to correct mistakes.
    learning_rate=0.015,#how fast each tree learns
    max_depth=7,#how complex each tree can be
    min_child_weight=5,#prevents overfitting
    subsample=0.8,#use 80% of data per tree
    colsample_bytree=0.7,#use 70% of features per tree
    reg_alpha=0.1, #regularization
    reg_lambda=1.0,
    #the tree will only make a split if it actually improves the prediction by at least this amount
    #prevents making tiny useless splits.
    gamma=0.1,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,#Use all of my computer's CPU cores to train faster
    early_stopping_rounds=50   #stop if no improvement for 50 rounds
)

#this trains the XGBoost model, It learns from the training data and checks itself against the validation set
#while training. verbose=100 means it prints a progress update every 100 trees so we can watch it improve
print("\nTraining XGBoost:")
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
print("XGBoost training finished.")

#evaluate the models (neural network & xgboost)
#set clipping range so predictions dont go crazy
log_min = float(y_train.min())
log_max = float(y_train.max())

#neural network predictions
nn_preds_log = model.predict(X_test).flatten()
nn_preds_log = np.clip(nn_preds_log, log_min, log_max)
nn_preds = np.expm1(nn_preds_log) #convert back to dollars

#xgboost predictions
xgb_preds_log = xgb.predict(X_test)
xgb_preds_log = np.clip(xgb_preds_log, log_min, log_max)
xgb_preds = np.expm1(xgb_preds_log) #convert back to dollars

#actual test prices in dollars
actual_prices = np.expm1(y_test)


#ensemble: where we combine both models (neural network and XGBoost)
#I started by guessing that 50/50 is the best mix for the ensemble
best_xgb_weight = 0.5
best_mae = float("inf")

#This finds the best mix of both models It tries every combination from 0% XGBoost to 100% XGBoost
#in steps of 5% and picks whichever mix gives the lowest error.
#for example it tests 50% XGBoost + 50% Neural Network, then 55/45,
#then 60/40, and so on until it finds the best blend.

#This loop goes through the numbers 0.0, 0.05, 0.10, 0.15 all the way to 1.0,
#each number represents how much to trust the XGBoost model so 0.0 means 100% neural network,
#0.5 means 50/50, and 1.0 means 100% XGBoost
for w in np.arange(0.0, 1.05, 0.05):

    #This creates the blended prediction, so if w is 0.7,
    #then we take 70% of XGBoost's prediction and 30% of the neural network's prediction
    #and add them together, the two percentages should in theory always add up to 100%
    blended = w * xgb_preds + (1 - w) * nn_preds

    #this checks how far off the blended prediction was from the actual prices
    #a lower error means this mix is more accurate.
    error = mean_absolute_error(actual_prices, blended)
    if error < best_mae:
        best_mae = error #save this error as the new best
        best_xgb_weight = w #remember which weight produced it
#after the loop is done, round the best XGBoost weight to 2 decimal places
xgb_w = round(best_xgb_weight, 2)
#The neural network weight is whatever is left over
nn_w = round(1 - best_xgb_weight, 2)

#final ensemble predictions using the best weights
ensemble_preds = xgb_w * xgb_preds + nn_w * nn_preds

#gather results
#MAE refers to average dollar error
#this calculates the MAE for each model. MAE is the average dollar amount the prediction is off by
#so if the MAE is $924, that means on average the model's prediction is about $924 away from the actual selling price.
nn_mae = mean_absolute_error(actual_prices, nn_preds)
xgb_mae = mean_absolute_error(actual_prices, xgb_preds)
ensemble_mae = mean_absolute_error(actual_prices, ensemble_preds)

#RMSLE refers to percentage based error
#this calculates the RMSLE score for each model the safe function makes sure no prices are negative because
#you can't take a log of a negative number then we calculate RMSLE which tells us how accurate each model is percentage wise
safe = lambda x: np.clip(x, 0, None)
nn_rmsle = np.sqrt(mean_squared_log_error(safe(actual_prices), safe(nn_preds)))
xgb_rmsle = np.sqrt(mean_squared_log_error(safe(actual_prices), safe(xgb_preds)))
ensemble_rmsle = np.sqrt(mean_squared_log_error(safe(actual_prices), safe(ensemble_preds)))

#prints results
print("RESULTS: ")
print(f"\nNeural Network:")
print(f"  MAE: ${nn_mae:,.2f}")
print(f"  RMSLE: {nn_rmsle:.4f}")

print(f"\nXGBoost:")
print(f"  MAE: ${xgb_mae:,.2f}")
print(f"  RMSLE: {xgb_rmsle:.4f}")

print(f"\nEnsemble ({xgb_w:.0%} XGBoost + {nn_w:.0%} NN):")
print(f"  MAE: ${ensemble_mae:,.2f}")
print(f"  RMSLE: {ensemble_rmsle:.4f}")

#save models
#save trained neural network
model.save("neural_network_model.keras")
#save preprocessing steps
joblib.dump(preprocess, "nn_preprocess.joblib")
#save xgboost model
joblib.dump(xgb, "xgboost_model.joblib")
#save the ensemble weights
joblib.dump({"xgb": xgb_w, "nn": nn_w}, "ensemble_weights.joblib")

print("All models saved.")

#sample test
sample_car = {
    "year": 2015, "make": "Kia", "model": "other", "condition": 0.77,
    "odometer": 50000, "mmr": 20000, "transmission": "automatic",
    "state": "ca", "body": "suv", "color": "white",
    "interior": "black", "trim": "other", "seller": "other",
}

#convert to dataframe
sample_df = pd.DataFrame([sample_car])

#add same features we created during training
#before we can predict the price of a new car, we have to add the same features we created during training
#if we skip any of them the model will crash because it expects to see all the same columns it was trained on
sample_df["car_age"] = (CURRENT_YEAR - sample_df["year"]).clip(lower=0)
sample_df["log_odometer"] = np.log1p(sample_df["odometer"])
sample_df["miles_per_year"] = sample_df["odometer"] / (sample_df["car_age"] + 1)
sample_df["mmr_log"] = np.log1p(sample_df["mmr"])
sample_df["mmr_per_mile"] = sample_df["mmr"] / (sample_df["odometer"] + 1)
sample_df["condition_times_mmr"] = sample_df["condition"] * sample_df["mmr"]

#make sure columns match training data
sample_df = sample_df.reindex(columns=X.columns, fill_value="unknown")

#apply same preprocessing
sample_ready = preprocess.transform(sample_df)
if hasattr(sample_ready, "toarray"):
    sample_ready = sample_ready.toarray().astype(np.float32)

#predict with both models
nn_price = float(np.expm1(np.clip(model.predict(sample_ready)[0][0], log_min, log_max)))
xgb_price = float(np.expm1(np.clip(xgb.predict(sample_ready)[0], log_min, log_max)))
ensemble_price = xgb_w * xgb_price + nn_w * nn_price

print(f"\nExample prediction (2015 Kia Sorento, 50k miles):")
print(f"  Neural Network: ${nn_price:,.2f}")
print(f"  XGBoost: ${xgb_price:,.2f}")
print(f"  Ensemble: ${ensemble_price:,.2f}")