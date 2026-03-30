import pandas as pd
import yfinance as yf
import pandas_datareader.data as pdr
from datetime import timedelta
import os

print("1. Đang tải dữ liệu gốc...")
train_df = pd.read_csv("data/gold_train.csv", index_col="Date", parse_dates=True)
test_df = pd.read_csv("data/gold_test.csv", index_col="Date", parse_dates=True)

# Lấy khoảng thời gian để tải dữ liệu
start_date = min(train_df.index.min(), test_df.index.min()) - timedelta(days=7)
end_date = max(train_df.index.max(), test_df.index.max()) + timedelta(days=7)

print(f"-> Cần tải dữ liệu vĩ mô từ {start_date.date()} đến {end_date.date()}")

print("\n2. Đang kéo dữ liệu Chỉ số USD (DXY) từ Yahoo Finance...")
dxy_data = yf.download("DX-Y.NYB", start=start_date, end=end_date)

# --- XỬ LÝ LỖI YFINANCE UPDATE Ở ĐÂY ---
if isinstance(dxy_data.columns, pd.MultiIndex):
    dxy = dxy_data['Close'].iloc[:, 0] # Lấy cột giá trị cốt lõi nếu bị MultiIndex
else:
    dxy = dxy_data['Close']
dxy.name = 'DXY'
# ---------------------------------------

print("3. Đang kéo dữ liệu Lãi suất FED (DFF) từ FRED...")
fed_data = pdr.DataReader('DFF', 'fred', start_date, end_date)
fed_rate = fed_data['DFF'].rename('FED_Rate')

print("\n4. Đang xử lý và ghép nối dữ liệu...")
full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
exog_df = pd.DataFrame(index=full_date_range)

exog_df = exog_df.join(dxy).join(fed_rate)

# Lấp đầy các ngày lễ/cuối tuần bằng giá trị của ngày giao dịch trước đó
exog_df.ffill(inplace=True)
exog_df.bfill(inplace=True) 

# Ghép vào tập train/test
train_exog = train_df.join(exog_df)
test_exog = test_df.join(exog_df)

# Xóa NaN lần cuối
train_exog.ffill(inplace=True)
test_exog.ffill(inplace=True)

print("\n5. Đang lưu file dữ liệu mới...")
train_exog.to_csv("data/gold_train_exog.csv")
test_exog.to_csv("data/gold_test_exog.csv")

print("\n✅ HOÀN TẤT! Đã lưu thành công 2 file mới.")

print("\nXem thử 5 dòng đầu của tập Train mới:")
print(train_exog[['Close', 'DXY', 'FED_Rate']].head())