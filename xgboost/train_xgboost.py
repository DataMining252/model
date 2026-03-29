from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

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

# Feature engineering
## Lag features
for i in range(1, 15):
        df[f'lag_{i}'] = df['Close'].shift(i)

## Rolling statistics
df['rolling_mean_7'] = df['Close'].rolling(7).mean()
df['rolling_mean_30'] = df['Close'].rolling(30).mean()
df['rolling_std_7'] = df['Close'].rolling(7).std()

df['rolling_max_7'] = df['Close'].rolling(7).max()
df['rolling_min_7'] = df['Close'].rolling(7).min()

df['day_of_week'] = df['Date'].dt.dayofweek   # 0=Mon, 6=Sun
df['month'] = df['Date'].dt.month

df["price_range_7"] = df["Close"].rolling(7).max() - df["Close"].rolling(7).min()
df["trend_7"] = df["Close"] - df["Close"].shift(7)


## Return
df['return'] = df['Close'].pct_change()

# Multi-step targets
for i in range(1, 8):
        df[f'target_{i}'] = df['Close'].pct_change(i).shift(-i)

df = df.dropna()

# Features
feature_cols = [
    col for col in df.columns
    if col.startswith('lag_') 
    or 'rolling' in col 
    or col in ['return', 'day_of_week', 'month', 'price_range_7', 'trend_7']
]
feature_cols.append('Close')

# Train test split
split_idx = int(len(df) * 0.8)

train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

X_train.to_csv(DATA_DIR / "X_train.csv", index=False)
X_test.to_csv(DATA_DIR / "X_test.csv", index=False)

# y_train.to_csv(DATA_DIR / "y_train.csv", index=False)
# y_test.to_csv(DATA_DIR / "y_test.csv", index=False)

# Train model
model_dict = {}
preds = pd.DataFrame(index=X_test.index)

print("Training models...\n")
for i in range(1, 8):
        print(f"Training model for day +{i}")
        y_train = train_df[f'target_{i}']
        y_test = test_df[f'target_{i}']

        model = XGBRegressor(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=42,
                n_jobs=-1
        )

        model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
        )
        y_pred = model.predict(X_test)

        model_dict[f'day_{i}'] = model

        preds[f'pred_{i}'] = y_pred

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Day +{i} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}\n")

# ======================
# SAVE MODELS
# ======================
print("Saving models...")

for i in range(1, 8):
    joblib.dump(model_dict[f'day_{i}'], MODEL_DIR / f"xgb_day_{i}.pkl")

print("All models saved!")

last_close = df['Close'].iloc[split_idx - 1]

price_preds = []

for i in range(1, 8):
    returns = preds[f'pred_{i}'].values
    
    # convert return -> price
    prices = last_close * (1 + returns)
    price_preds.append(prices)

price_preds = np.array(price_preds).T

import matplotlib.pyplot as plt
plt.figure(figsize=(14,8))
last_close_series = test_df['Close'].shift(0).values
for i in range(1, 8):
    plt.subplot(4, 2, i)

    actual = test_df[f'target_{i}'].values
    pred = preds[f'pred_{i}'].values

    actual_price = last_close_series * (1 + actual)
    pred_price = last_close_series * (1 + pred)

    plt.plot(actual_price, label="Actual")
    plt.plot(pred_price, label="Pred")

    plt.title(f"Day +{i}")

plt.tight_layout()
plt.show()