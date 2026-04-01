import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima import auto_arima

# Tự động tạo thư mục models nếu chưa có
os.makedirs("models", exist_ok=True)

# =========================
# Load data
# =========================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# statistical/data/...
TRAIN_PATH = BASE_DIR / "statistical" / "data" / "gold_train.csv"
TEST_PATH  = BASE_DIR / "statistical" / "data" / "gold_test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

train_df["Date"] = pd.to_datetime(train_df["Date"])
test_df["Date"] = pd.to_datetime(test_df["Date"])

train_df.set_index("Date", inplace=True)
test_df.set_index("Date", inplace=True)

train = train_df["Close"]
test = test_df["Close"]

train.index = pd.DatetimeIndex(train.index)
test.index = pd.DatetimeIndex(test.index)

train = train.asfreq("D")
test = test.asfreq("D")

train = train.ffill()
test = test.ffill()

# --- CHỈ DỰ BÁO 1 TUẦN (7 NGÀY) ---
forecast_steps = 7
test_1_week = test.iloc[:forecast_steps]

# =========================
# ADF Test
# =========================

result = adfuller(train)

print("ADF Statistic:", result[0])
print("p-value:", result[1])

# =========================
# BASELINE MODEL: ARIMA(1,1,1)
# =========================

print("\n===== BASELINE ARIMA(1,1,1) =====")

arima_model = ARIMA(train, order=(1,1,1))
arima_fit = arima_model.fit()

print(arima_fit.summary())

# Sửa lại số bước dự báo và khớp index của 7 ngày
forecast_arima = arima_fit.forecast(steps=forecast_steps)
forecast_arima = pd.Series(forecast_arima.values, index=test_1_week.index)

rmse_arima = np.sqrt(mean_squared_error(test_1_week, forecast_arima))
mae_arima = mean_absolute_error(test_1_week, forecast_arima)

print("ARIMA RMSE:", rmse_arima)
print("ARIMA MAE:", mae_arima)


# =========================
# AUTO ARIMA
# =========================

print("\n===== AUTO ARIMA SEARCH =====")

auto_model = auto_arima(
    train,
    seasonal=False,
    trace=True,
    stepwise=True,
    suppress_warnings=True
)

print(auto_model.summary())

# Train best ARIMA
best_order = auto_model.order
print("Best ARIMA order:", best_order)

best_arima = ARIMA(train, order=best_order)
best_fit = best_arima.fit()

# Sửa lại số bước dự báo và khớp index của 7 ngày
forecast_auto = best_fit.forecast(steps=forecast_steps)
forecast_auto = pd.Series(forecast_auto.values, index=test_1_week.index)

# Evaluate Auto ARIMA
rmse_auto = np.sqrt(mean_squared_error(test_1_week, forecast_auto))
mae_auto = mean_absolute_error(test_1_week, forecast_auto)

print("\nAUTO ARIMA RMSE:", rmse_auto)
print("AUTO ARIMA MAE:", mae_auto)


# =========================
# Save best model
# =========================

with open("models/arima_gold_model.pkl", "wb") as f:
    pickle.dump(best_fit, f)

print("Best model saved to models/arima_gold_model.pkl")


# =========================
# Comparison
# =========================

print("\n===== MODEL COMPARISON =====")

print("ARIMA(1,1,1) RMSE:", rmse_arima)
print("Auto ARIMA RMSE:", rmse_auto)

print("ARIMA(1,1,1) MAE:", mae_arima)
print("Auto ARIMA MAE:", mae_auto)


# =========================
# Plot
# =========================

plt.figure(figsize=(12,6))

# MẸO: Chỉ vẽ 90 ngày cuối của tập train để dễ nhìn đoạn nối tiếp 1 tuần
train_plot = train.iloc[-90:]

plt.plot(train_plot.index, train_plot, label="Train (90 ngày cuối)")
plt.plot(test_1_week.index, test_1_week, label="Test (1 Tuần thực tế)", marker='o')

plt.plot(test_1_week.index, forecast_arima, label="ARIMA(1,1,1)", linestyle='--')
plt.plot(test_1_week.index, forecast_auto, label="Auto ARIMA", linestyle='--')

plt.legend()
plt.title("ARIMA vs Auto ARIMA Forecast (1 Tuần)")
plt.grid(True)
plt.show()