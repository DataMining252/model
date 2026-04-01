import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
from pathlib import Path

# Base dir = thư mục chứa file train_xgboost.py
BASE_DIR = Path(__file__).resolve().parent

# ../raw/final_dataset.csv
DATA_PATH = BASE_DIR.parent / "raw" / "final_dataset.csv"

# ./models/xgb_model.pkl
MODEL_PATH = BASE_DIR / "models" / "xgb_model.pkl"

# ======================
# LOAD DATA
# ======================
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# ======================
# FEATURE ENGINEERING
# ======================
def create_features(df):
    df = df.copy()

    # Lag
    for lag in range(1, 15):
        df[f'lag_{lag}'] = df['Close'].shift(lag)

    # Rolling
    df['ma_7'] = df['Close'].rolling(7).mean()
    df['std_7'] = df['Close'].rolling(7).std()

    # Return
    df['return_1'] = df['Close'].pct_change(1)
    df['return_7'] = df['Close'].pct_change(7)

    # Momentum
    df['momentum_7'] = df['Close'] - df['Close'].shift(7)

    return df

df = create_features(df)

# ======================
# TARGET (RETURN 7 DAYS)
# ======================
for i in range(1, 8):
    df[f'target_{i}'] = df['Close'].pct_change(i).shift(-i)

df = df.dropna()

# ======================
# FEATURES
# ======================
features = [col for col in df.columns if
            'lag_' in col or 'ma_' in col or 'std_' in col
            or 'return_' in col or 'momentum_' in col]

# thêm macro features nếu có
macro_cols = ['DXY', 'SP500', 'OIL', 'INTEREST_RATE', 'CPI']
for col in macro_cols:
    if col in df.columns:
        features.append(col)

# ======================
# TRAIN
# ======================
models = {}

for i in range(1, 8):
    print(f"Training Day +{i}")

    X = df[features]
    y = df[f'target_{i}']

    split = int(len(df) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)
    models[i] = model

# ======================
# SAVE MODEL
# ======================
MODEL_PATH = Path(MODEL_PATH)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump({
    "models": models,
    "features": features
}, MODEL_PATH)

print("Saved model!")

# ======================
# EVALUATION (PRICE LEVEL)
# ======================
fig, axes = plt.subplots(4, 2, figsize=(15, 12))
axes = axes.flatten()

for i in range(1, 8):
    model = models[i]

    X = df[features]
    y_return = df[f'target_{i}']

    split = int(len(df) * 0.8)
    X_test = X[split:]
    y_return_test = y_return[split:]

    # predict return
    y_pred_return = model.predict(X_test)

    # convert to price
    base_prices = df['Close'].shift(1)[split:]  # giá hôm trước

    y_true_price = base_prices * (1 + y_return_test)
    y_pred_price = base_prices * (1 + y_pred_return)

    # metrics
    mae = mean_absolute_error(y_true_price, y_pred_price)
    rmse = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
    r2 = r2_score(y_true_price, y_pred_price)

    print(f"\nDay +{i}")
    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2  : {r2:.4f}")

    ax = axes[i-1]
    ax.plot(y_true_price.values[-100:], label="Actual")
    ax.plot(y_pred_price.values[-100:], label="Pred")

    ax.set_title(f"Day +{i} | R2={r2:.2f}")
    ax.grid()

# remove extra plot
fig.delaxes(axes[7])

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(4, 2, figsize=(12, 12))
axes = axes.flatten()

for i in range(1, 8):
    model = models[i]

    X = df[features]
    y_return = df[f'target_{i}']

    split = int(len(df) * 0.8)
    X_test = X[split:]
    y_return_test = y_return[split:]

    # predict return
    y_pred_return = model.predict(X_test)

    # convert to price
    base_prices = df['Close'].shift(1)[split:]

    y_true_price = base_prices * (1 + y_return_test)
    y_pred_price = base_prices * (1 + y_pred_return)

    ax = axes[i-1]

    # scatter
    ax.scatter(y_true_price, y_pred_price, alpha=0.5)

    # đường y = x
    min_val = min(y_true_price.min(), y_pred_price.min())
    max_val = max(y_true_price.max(), y_pred_price.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle='--')

    ax.set_title(f"Day +{i}")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid()

# xóa ô thừa
fig.delaxes(axes[7])

plt.tight_layout()
plt.show()