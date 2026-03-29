import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ======================
# CONFIG
# ======================
DATA_PATH = "../raw/final_dataset.csv"
MODEL_PATH = "./models/rf_model.pkl"

WINDOW_SIZE = 1500
STEP = 150
HORIZON = 7

# ======================
# LOAD DATA
# ======================
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.sort_values("Date")

# ======================
# FEATURE ENGINEERING
# ======================

# Log return + SCALE
df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
df["log_return_scaled"] = df["log_return"] * 100

# Clip outlier
df["log_return_scaled"] = df["log_return_scaled"].clip(-5, 5)

# Lag
for lag in [1, 2, 3, 7, 14]:
    df[f"log_return_lag_{lag}"] = df["log_return_scaled"].shift(lag)
    df[f"close_lag_{lag}"] = df["Close"].shift(lag)

# Trend + momentum
df["ma_10"] = df["Close"].rolling(10).mean()
df["ma_20"] = df["Close"].rolling(20).mean()
df["trend"] = df["ma_10"] - df["ma_20"]
df["momentum"] = df["Close"] - df["Close"].shift(10)

# Volatility
df["volatility"] = df["log_return_scaled"].rolling(10).std()

# Short-term signals
df["return_2"] = df["log_return_scaled"].rolling(2).mean()
df["return_5"] = df["log_return_scaled"].rolling(5).mean()

# ======================
# TARGET
# ======================
for i in range(1, HORIZON + 1):
    df[f"target_{i}"] = df["log_return_scaled"].shift(-i)

df = df.dropna()

# ======================
# FEATURES
# ======================
targets = [f"target_{i}" for i in range(1, HORIZON + 1)]
features = [col for col in df.columns if col not in ["Date"] and not col.startswith("target")]

# ======================
# WALK-FORWARD
# ======================
all_preds = []
all_actuals = []

for start in range(0, len(df) - WINDOW_SIZE - HORIZON, STEP):
    print(f"Processing window {start}/{len(df)}")

    train_df = df.iloc[start : start + WINDOW_SIZE]
    test_df  = df.iloc[start + WINDOW_SIZE : start + WINDOW_SIZE + STEP]

    X_train = train_df[features]
    y_train = train_df[targets]

    X_test = test_df[features]
    y_test = test_df[targets]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    preds = pd.DataFrame(preds, columns=targets, index=y_test.index)

    all_preds.append(preds)
    all_actuals.append(y_test)

# ======================
# CONCAT
# ======================
preds_df = pd.concat(all_preds)
actual_df = pd.concat(all_actuals)

print("Walk-forward done!")

# ======================
# SAVE MODEL
# ======================
os.makedirs("models", exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to {MODEL_PATH}")

# ======================
# METRICS
# ======================
print("\n=== METRICS ===")

for i in range(1, HORIZON + 1):
    mae = mean_absolute_error(actual_df[f"target_{i}"], preds_df[f"target_{i}"])
    rmse = np.sqrt(mean_squared_error(actual_df[f"target_{i}"], preds_df[f"target_{i}"]))

    print(f"Day +{i}: MAE={mae:.4f}, RMSE={rmse:.4f}")

# Direction accuracy
direction_acc = (np.sign(preds_df) == np.sign(actual_df)).mean()
print("\nDirection accuracy:")
print(direction_acc)

# ======================
# SMOOTH FUNCTION
# ======================
def smooth(x, window=10):
    return pd.Series(x).rolling(window).mean()

# ======================
# PLOT
# ======================
plt.figure(figsize=(14, 12))

for i in range(1, HORIZON + 1):
    plt.subplot(4, 2, i)

    actual = actual_df[f"target_{i}"].values
    pred = preds_df[f"target_{i}"].values

    plt.plot(smooth(actual), label="Actual (smooth)")
    plt.plot(smooth(pred), label="Predicted (smooth)")

    plt.title(f"Day +{i}")
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()

# ======================
# SCATTER
# ======================
plt.close("all")
plt.figure(figsize=(14, 12))

for i in range(1, HORIZON + 1):
    plt.subplot(4, 2, i)

    actual = actual_df[f"target_{i}"]
    pred = preds_df[f"target_{i}"]

    plt.scatter(actual, pred, alpha=0.5)

    min_val = min(actual.min(), pred.min())
    max_val = max(actual.max(), pred.max())
    plt.plot([min_val, max_val], [min_val, max_val])

    plt.title(f"Scatter Day +{i}")
    plt.grid()

plt.tight_layout()
plt.show()