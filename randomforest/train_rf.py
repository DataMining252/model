import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ======================
# CONFIG
# ======================
DATA_PATH = "../raw/final_dataset.csv"
MODEL_PATH = "./models/rf_recursive_7day.pkl"

WINDOW_SIZE = 1500
STEP = 150
HORIZON = 7

# ======================
# LOAD DATA
# ======================
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# ======================
# FEATURE ENGINEERING
# ======================
df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
df["log_return_scaled"] = df["log_return"] * 100
df["log_return_scaled"] = df["log_return_scaled"].clip(-5, 5)

for lag in [1,2,3,7,14]:
    df[f"log_return_lag_{lag}"] = df["log_return_scaled"].shift(lag)
    df[f"close_lag_{lag}"] = df["Close"].shift(lag)

df["ma_10"] = df["Close"].rolling(10).mean()
df["ma_20"] = df["Close"].rolling(20).mean()
df["trend"] = df["ma_10"] - df["ma_20"]
df["momentum"] = df["Close"] - df["Close"].shift(10)
df["volatility"] = df["log_return_scaled"].rolling(10).std()
df["return_2"] = df["log_return_scaled"].rolling(2).mean()
df["return_5"] = df["log_return_scaled"].rolling(5).mean()

df = df.dropna().reset_index(drop=True)

features_base = [col for col in df.columns if col not in ["Date","log_return","log_return_scaled","Close"]]

# ======================
# WALK-FORWARD + RECURSIVE 7-DAY FORECAST
# ======================
all_preds_horizon = []
all_actuals_horizon = []

for start in range(0, len(df) - WINDOW_SIZE - HORIZON, STEP):
    print(f"Processing window {start}/{len(df)}")

    train_df = df.iloc[start:start+WINDOW_SIZE]
    test_df  = df.iloc[start+WINDOW_SIZE:start+WINDOW_SIZE+STEP]

    X_train = train_df[features_base]
    y_train = train_df["log_return_scaled"]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # ======================
    # Recursive 7-day forecast
    # ======================
    for i in range(len(test_df)):
        # start price = actual price at step i
        price = test_df["Close"].iloc[i]
        preds_row = []
        actual_row = []
        for h in range(HORIZON):
            row_features = test_df.iloc[i][features_base].copy()
            # update all close_lag features using last forecast
            for lag in [1,2,3,7,14]:
                row_features[f"close_lag_{lag}"] = price

            # predict log return
            pred_log_return = model.predict(pd.DataFrame([row_features]))[0]
            # update price
            price = price * np.exp(pred_log_return/100)
            preds_row.append(price)

            # actual price Day+h
            idx_actual = i + h
            if idx_actual < len(test_df):
                actual_row.append(test_df.iloc[idx_actual]["Close"])
            else:
                actual_row.append(np.nan)

        all_preds_horizon.append(preds_row)
        all_actuals_horizon.append(actual_row)

# ======================
# CONVERT TO DATAFRAME
# ======================
columns_horizon = [f"Day+{i}" for i in range(1,HORIZON+1)]
price_preds_horizon = pd.DataFrame(all_preds_horizon, columns=columns_horizon)
price_actuals_horizon = pd.DataFrame(all_actuals_horizon, columns=columns_horizon)

# ======================
# SAVE MODEL
# ======================
os.makedirs("models", exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
print(f"Model saved to {MODEL_PATH}")

# ======================
# PLOT 7-DAY PRICE
# ======================
plt.figure(figsize=(14,12))
for i in range(HORIZON):
    plt.subplot(4,2,i+1)
    plt.plot(price_actuals_horizon[f"Day+{i+1}"].values, label="Actual Price")
    plt.plot(price_preds_horizon[f"Day+{i+1}"].values, label="Predicted Price")
    plt.title(f"Price Forecast Day +{i+1}")
    plt.legend()
    plt.grid()
plt.tight_layout()
plt.show()

# ======================
# SCATTER 7-DAY
# ======================
plt.figure(figsize=(14,12))
for i in range(HORIZON):
    plt.subplot(4,2,i+1)
    actual = price_actuals_horizon[f"Day+{i+1}"].values
    pred   = price_preds_horizon[f"Day+{i+1}"].values
    plt.scatter(actual, pred, alpha=0.5)
    min_val = min(np.nanmin(actual), np.nanmin(pred))
    max_val = max(np.nanmax(actual), np.nanmax(pred))
    plt.plot([min_val,max_val],[min_val,max_val], color="red")
    plt.title(f"Scatter Day +{i+1}")
    plt.grid()
plt.tight_layout()
plt.show()