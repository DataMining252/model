import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# ==========================================
# CẤU HÌNH TRANG STREAMLIT
# ==========================================
st.set_page_config(page_title="Dashboard Khai Phá Dữ Liệu Giá Vàng", page_icon="🏆", layout="wide")

st.title("🏆 Dashboard So Sánh Các Mô Hình Dự Báo Giá Vàng (7 Ngày)")


# ==========================================
# HÀM LOAD DỮ LIỆU & TÍNH TOÁN
# ==========================================
@st.cache_data
def load_and_predict():
    forecast_steps = 7
    results = {}
    
    # 1. TẢI DỮ LIỆU GỐC (Dùng chung để lấy Index)
    test_df = pd.read_csv("statistical/data/gold_test.csv", index_col="Date", parse_dates=True).asfreq("D").ffill()
    train_df = pd.read_csv("statistical/data/gold_train.csv", index_col="Date", parse_dates=True).asfreq("D").ffill()
    
    y_train = train_df["Close"]
    y_test = test_df["Close"].iloc[:forecast_steps]
    forecast_index = y_test.index
    
    results['y_train'] = y_train
    results['y_test'] = y_test
    
    # 2. DỰ BÁO ARIMA
    try:
        with open("models/arima_gold_model.pkl", "rb") as f:
            arima_model = pickle.load(f)
        arima_pred = arima_model.forecast(steps=forecast_steps)
        results['ARIMA'] = pd.Series(arima_pred.values, index=forecast_index)
    except Exception as e:
        results['ARIMA'] = None
        
    # 3. DỰ BÁO SARIMAX
    try:
        test_exog = pd.read_csv("statistical/data/gold_test_exog.csv", index_col="Date", parse_dates=True).asfreq("D").ffill()
        X_test_exog = test_exog[["DXY", "FED_Rate"]].iloc[:forecast_steps]
        
        with open("models/sarimax_gold_model.pkl", "rb") as f:
            sarimax_model = pickle.load(f)
        sarimax_pred = sarimax_model.forecast(steps=forecast_steps, exog=X_test_exog)
        results['SARIMAX'] = pd.Series(sarimax_pred.values, index=forecast_index)
    except Exception as e:
        results['SARIMAX'] = None
        
    # 4. DỰ BÁO RANDOM FOREST
    try:
        # Tự động tìm đường dẫn file test.csv (nếu bạn để ở ngoài hoặc trong statistical/data)
        rf_test_path = "test.csv"
        if os.path.exists("randomforest/data/test.csv"):
            rf_test_path = "randomforest/data/test.csv"
        elif os.path.exists("randomforest/data/test.csv"):
            rf_test_path = "randomforest/data/test.csv"

        df_test_rf = pd.read_csv(rf_test_path)
        
        # Lọc bỏ các cột không phải là Feature
        target_cols = [f"target_{i}" for i in range(1, 8)]
        drop_cols = ["Date"] + target_cols
        X_test_rf = df_test_rf.drop(columns=drop_cols).iloc[[0]]
        
        with open("models/rf_gold_model.pkl", "rb") as f: 
            rf_model = pickle.load(f)
            
        # Dự báo 7 ngày từ dòng đầu tiên
        rf_pred = rf_model.predict(X_test_rf)[0]
        results['Random Forest'] = pd.Series(rf_pred, index=forecast_index)
    except Exception as e:
        results['Random Forest'] = None
        # Hiển thị lỗi ra giao diện để dễ gỡ nếu có
        st.sidebar.error(f"Lỗi khi tải Random Forest: {e}")
        
    return results

# ==========================================
# KHỞI CHẠY VÀ HIỂN THỊ
# ==========================================
data = load_and_predict()
y_test = data['y_test']
y_train = data['y_train']

st.header("1. So Sánh Sai Số (Metrics)")
# Tạo 3 cột để hiển thị thẻ điểm số
col1, col2, col3 = st.columns(3)

def show_metric(col, model_name, y_true, y_pred, color_hex):
    if y_pred is not None:
        # Ép phẳng dữ liệu (.ravel()) để không bị lỗi [1, 7]
        y_t = np.array(y_true).ravel()
        y_p = np.array(y_pred).ravel()
        
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        mae = mean_absolute_error(y_t, y_p)
        
        col.markdown(f"### <span style='color:{color_hex}'>{model_name}</span>", unsafe_allow_html=True)
        col.metric(label="📉 RMSE", value=f"{rmse:.2f} USD")
        col.metric(label="📊 MAE", value=f"{mae:.2f} USD")
    else:
        col.markdown(f"### <span style='color:{color_hex}'>{model_name}</span>", unsafe_allow_html=True)
        col.warning("Chưa có Model")

show_metric(col1, "1. ARIMA", y_test, data['ARIMA'], "#1f77b4")
show_metric(col2, "2. SARIMAX", y_test, data['SARIMAX'], "#2ca02c")
show_metric(col3, "3. Random Forest", y_test, data['Random Forest'], "#d62728")

st.write("---")
st.header("2. Đồ Thị Tổng Hợp Các Đường Dự Báo")

# Vẽ biểu đồ tổng hợp
fig, ax = plt.subplots(figsize=(14, 6))

# Cắt 60 ngày cuối của Train để biểu đồ zoom sát vào đoạn dự báo
train_plot = y_train.iloc[-60:]

# Vẽ Dữ liệu gốc
ax.plot(train_plot.index, train_plot, label="Train (60 ngày cuối)", color='gray', alpha=0.5, linewidth=2)
ax.plot(y_test.index, y_test, label="Thực tế (Test - 7 ngày)", color='black', marker='o', linewidth=3)

# Vẽ các đường dự báo
if data['ARIMA'] is not None:
    ax.plot(y_test.index, data['ARIMA'], label="ARIMA", color='#1f77b4', linestyle='--', linewidth=2)
if data['SARIMAX'] is not None:
    ax.plot(y_test.index, data['SARIMAX'], label="SARIMAX (Có DXY & FED)", color='#2ca02c', linestyle='-.', linewidth=2.5)
if data['Random Forest'] is not None:
    ax.plot(y_test.index, data['Random Forest'], label="Random Forest", color='#d62728', linestyle=':', linewidth=2.5, marker='X')

ax.set_title("So sánh trực quan biến động thực tế và dự báo của 3 mô hình", fontsize=15)
ax.set_ylabel("Giá Đóng Cửa (USD)", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.4)

st.pyplot(fig)

