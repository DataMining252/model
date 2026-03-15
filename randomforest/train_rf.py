import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Config
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DATASET_PATH = BASE_DIR.parent / 'gold_with_season.csv'

# Load data set
df = pd.read_csv(DATASET_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
df["Date"] = pd.to_datetime(df["Date"])

df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Month"] = df["Date"].dt.month
df["Quarter"] = df["Date"].dt.quarter

# Feature engineering
## Lag feature
df["Close_lag1"] = df["Close"].shift(1)
df["Close_lag2"] = df["Close"].shift(2)
df["Close_lag3"] = df["Close"].shift(3)
df["Close_lag7"] = df["Close"].shift(7)
df["Close_lag14"] = df["Close"].shift(14)

## Technical indicators
df["MA7"] = df["Close"].rolling(7).mean()
df["MA14"] = df["Close"].rolling(14).mean()
df["STD7"] = df["Close"].rolling(7).std()

df["Return"] = df["Close"].pct_change()

# Create target
for i in range (1, 8):
    df[f"Target_{i}"] = df["Close"].shift(-i)

# Drop NA
df = df.dropna()

# Select features
features = [
    "Open",
    "High",
    "Low",
    "Volume",

    "Month",
    "Quarter",
    "DayOfWeek",

    "Close_lag1",
    "Close_lag2",
    "Close_lag3",
    "Close_lag7",
    "Close_lag14",

    "MA7",
    "MA14",
    "STD7",
    "Return",
]

X = df[features]

y = df[[f"Target_{i}" for i in range(1, 8)]]

# Train-test split
split = int(len(df) * 0.8)
X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

X_train.to_csv(DATA_DIR / "X_train.csv", index=False)
X_test.to_csv(DATA_DIR / "X_test.csv", index=False)

y_train.to_csv(DATA_DIR / "y_train.csv", index=False)
y_test.to_csv(DATA_DIR / "y_test.csv", index=False)

print("Train/Test datasets saved.")

# Train model
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Evaluate

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
mape = np.mean(np.abs((y_test - pred) / y_test)) * 100

print("===== Random Forest Evaluation =====")
print("MAE :", mae)
print("RMSE:", rmse)
print("MAPE:", mape)

# Save model
MODEL_PATH = MODEL_DIR / "rf_gold.pkl"
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("\nModel saved at:", MODEL_PATH)