import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima import auto_arima

# Tự động tạo thư mục models nếu chưa có
os.makedirs("models", exist_ok=True)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# statistical/data/...
TRAIN_EXOG_PATH = BASE_DIR / "statistical" / "data" / "gold_train_exog.csv"
TEST_EXOG_PATH  = BASE_DIR / "statistical" / "data" / "gold_test_exog.csv"

# =========================
# 1. Load dữ liệu MỚI (có chứa biến ngoại sinh)
# =========================
print("Đang tải dữ liệu...")
# Lưu ý: Đổi tên file thành file _exog mà chúng ta vừa tạo
train_df = pd.read_csv(TRAIN_EXOG_PATH)
test_df  = pd.read_csv(TEST_EXOG_PATH)

train_df["Date"] = pd.to_datetime(train_df["Date"])
test_df["Date"] = pd.to_datetime(test_df["Date"])

train_df.set_index("Date", inplace=True)
test_df.set_index("Date", inplace=True)

# Đặt tần suất ngày và lấp đầy NaN
train_df = train_df.asfreq("D").ffill()
test_df = test_df.asfreq("D").ffill()

# --- CHỈ DỰ BÁO 1 TUẦN (7 NGÀY) ---
forecast_steps = 7
test_1_week = test_df.iloc[:forecast_steps]

# Tách biến mục tiêu (Close) và biến ngoại sinh (DXY, FED_Rate)
y_train = train_df["Close"]
X_train = train_df[["DXY", "FED_Rate"]]

y_test = test_1_week["Close"]
X_test = test_1_week[["DXY", "FED_Rate"]]

# =========================
# 2. Auto ARIMA tìm bậc SARIMAX tối ưu
# =========================
print("\n===== AUTO ARIMA TÌM BẬC SARIMAX =====")
# Truyền thêm X_train vào hàm
sarima_auto = auto_arima(
    y=y_train,
    X=X_train,                # ĐÂY LÀ ĐIỂM KHÁC BIỆT: Nạp biến ngoại sinh vào
    seasonal=True,
    m=5,                      # chu kỳ tuần (5 ngày giao dịch)
    trace=True,
    stepwise=True,            # Vẫn để True cho chạy nhanh, có thể đổi False sau
    suppress_warnings=True
)

print(sarima_auto.summary())
sarima_order = sarima_auto.order
sarima_seasonal_order = sarima_auto.seasonal_order
print(f"Bậc SARIMA tối ưu: {sarima_order} x {sarima_seasonal_order}")

# =========================
# 3. Huấn luyện mô hình SARIMAX
# =========================
print("\n===== HUẤN LUYỆN SARIMAX =====")
sarima_model = SARIMAX(
    endog=y_train,
    exog=X_train,             # Nạp biến ngoại sinh vào lúc Train
    order=sarima_order,
    seasonal_order=sarima_seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarima_fit = sarima_model.fit(disp=False)
print(sarima_fit.summary())

# =========================
# 4. Dự báo 1 Tuần
# =========================
# Lưu ý: Khi dự báo, bắt buộc phải cung cấp dữ liệu ngoại sinh của 7 ngày đó (X_test)
forecast_sarima_values = sarima_fit.forecast(steps=forecast_steps, exog=X_test)

print("\n===== KIỂM TRA DỰ BÁO =====")
forecast_sarima = pd.Series(forecast_sarima_values.values, index=y_test.index)

# =========================
# 5. Đánh giá mô hình
# =========================
rmse_sarima = np.sqrt(mean_squared_error(y_test, forecast_sarima))
mae_sarima = mean_absolute_error(y_test, forecast_sarima)
print(f"\nSARIMAX RMSE: {rmse_sarima:.4f}")
print(f"SARIMAX MAE: {mae_sarima:.4f}")

# =========================
# 6. Lưu mô hình
# =========================
with open("models/sarimax_gold_model.pkl", "wb") as f:
    pickle.dump(sarima_fit, f)
print("Model saved to models/sarimax_gold_model.pkl")

# =========================
# 7. Vẽ đồ thị
# =========================
plt.figure(figsize=(12,6))

# Cắt 90 ngày cuối của train để dễ nhìn đoạn nối tiếp
train_plot = y_train.iloc[-90:]

plt.plot(train_plot.index, train_plot, label="Train (90 ngày cuối)", color='blue', alpha=0.6)
plt.plot(y_test.index, y_test, label="Thực tế (7 ngày Test)", color='black', marker='o', linewidth=2)
plt.plot(y_test.index, forecast_sarima, label="SARIMAX Forecast", color='green', linestyle='--', linewidth=2)

plt.legend()
plt.title("SARIMAX Forecast (1 Tuần)")
plt.grid(True, alpha=0.4)
plt.show()