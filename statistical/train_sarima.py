import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima import auto_arima

# =========================
# 1. Load dữ liệu (giữ nguyên index datetime gốc)
# =========================
train_df = pd.read_csv("statistical/data/gold_train.csv")
test_df = pd.read_csv("statistical/data/gold_test.csv")

train_df["Date"] = pd.to_datetime(train_df["Date"])
test_df["Date"] = pd.to_datetime(test_df["Date"])

train_df.set_index("Date", inplace=True)
test_df.set_index("Date", inplace=True)

train = train_df["Close"]
test = test_df["Close"]

# Kiểm tra NaN gốc
print("NaN trong train gốc:", train.isnull().sum())
print("NaN trong test gốc:", test.isnull().sum())

# Nếu có NaN, xử lý (ví dụ: forward fill) nhưng không dùng asfreq
if train.isnull().any():
    train = train.ffill()
if test.isnull().any():
    test = test.ffill()

print("NaN trong train sau xử lý:", train.isnull().sum())
print("NaN trong test sau xử lý:", test.isnull().sum())

# =========================
# 2. Chuyển sang index số nguyên (để auto_arima và SARIMAX hoạt động ổn định)
# =========================
train_int = pd.Series(train.values, index=range(len(train)))
test_int = pd.Series(test.values, index=range(len(test)))

print(f"Độ dài train_int: {len(train_int)}")
print(f"Độ dài test_int: {len(test_int)}")
print(f"Độ dài test.index: {len(test.index)}")

# =========================
# 3. Auto ARIMA tìm bậc SARIMA tối ưu
# =========================
print("\n===== AUTO ARIMA (TÌM BẬC SARIMA) =====")
sarima_auto = auto_arima(
    train_int,
    seasonal=True,
    m=5,                      # chu kỳ tuần (5 ngày giao dịch)
    trace=True,
    stepwise=True,
    suppress_warnings=True
)

print(sarima_auto.summary())
sarima_order = sarima_auto.order
sarima_seasonal_order = sarima_auto.seasonal_order   # (P,D,Q,s)
print(f"Bậc SARIMA tối ưu: {sarima_order} x {sarima_seasonal_order}")

# =========================
# 4. Huấn luyện mô hình SARIMA
# =========================
print("\n===== HUẤN LUYỆN SARIMA =====")
sarima_model = SARIMAX(
    train_int,
    order=sarima_order,
    seasonal_order=sarima_seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarima_fit = sarima_model.fit(disp=False)
print(sarima_fit.summary())

# =========================
# 5. Dự báo
# =========================
forecast_sarima_values = sarima_fit.forecast(steps=len(test_int))

print("\n===== KIỂM TRA DỰ BÁO =====")
print("Số lượng NaN trong forecast_sarima_values:", np.isnan(forecast_sarima_values).sum())
print("Độ dài forecast_sarima_values:", len(forecast_sarima_values))
print("Độ dài test.index:", len(test.index))

# Kiểm tra khớp độ dài
if len(forecast_sarima_values) != len(test.index):
    print("Cảnh báo: Độ dài không khớp! Cắt index cho phù hợp.")
    min_len = min(len(forecast_sarima_values), len(test.index))
    forecast_sarima_values = forecast_sarima_values[:min_len]
    test_aligned = test.iloc[:min_len]
    forecast_index = test.index[:min_len]
else:
    test_aligned = test
    forecast_index = test.index

forecast_sarima = pd.Series(forecast_sarima_values, index=forecast_index)
print("NaN trong forecast_sarima sau khi gán index:", forecast_sarima.isnull().sum())

# =========================
# 6. Đánh giá mô hình
# =========================
if forecast_sarima.isnull().any() or test_aligned.isnull().any():
    print("Lỗi: Vẫn còn NaN trong dữ liệu dự báo hoặc test. Dừng lại.")
else:
    rmse_sarima = np.sqrt(mean_squared_error(test_aligned, forecast_sarima))
    mae_sarima = mean_absolute_error(test_aligned, forecast_sarima)
    print(f"\nSARIMA RMSE: {rmse_sarima:.4f}")
    print(f"SARIMA MAE: {mae_sarima:.4f}")

    # =========================
    # 7. Lưu mô hình
    # =========================
    with open("models/sarima_gold_model.pkl", "wb") as f:
        pickle.dump(sarima_fit, f)
    print("Model saved to models/sarima_gold_model.pkl")

    # =========================
    # 8. Vẽ đồ thị
    # =========================
    plt.figure(figsize=(12,6))
    plt.plot(train.index, train, label="Train")
    plt.plot(test_aligned.index, test_aligned, label="Test")
    plt.plot(forecast_sarima.index, forecast_sarima, label="Forecast")
    plt.legend()
    plt.title("SARIMA Forecast")
    plt.show()