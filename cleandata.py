import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

# CLEAN DATA
df = pd.read_csv("XAU_1d_data.csv", sep=";")
df.columns = df.columns.str.strip()

df["Date"] = pd.to_datetime(df["Date"], format="%Y.%m.%d %H:%M")

numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

df = df.sort_values("Date")

# Nếu muốn dùng cho time series
df = df.set_index("Date")
df2 = df.copy()

# DATA UNDERSTANDING
df.info()
df.describe()
df.isnull().sum()

# EDA TIME SERIES
df["Close"].plot(figsize=(12,5))
plt.title("XAU/USD Close Price")
plt.show()

# ROLLING MEAN AND ROLLING STD
rolling_mean = df["Close"].rolling(window=30).mean()
rolling_std = df["Close"].rolling(window=30).std()

plt.figure(figsize=(12,5))
plt.plot(df["Close"], label="Original")
plt.plot(rolling_mean, label="Rolling Mean")
plt.plot(rolling_std, label="Rolling Std")
plt.legend()
plt.show()

# SEASONALITY
result = seasonal_decompose(df["Close"], model="multiplicative", period=365)
result.plot()
plt.show()

# ADF TEST
adf_result = adfuller(df["Close"])
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])

# KPSS TEST
kpss_result = kpss(df["Close"], regression='c')
print("KPSS Statistic:", kpss_result[0])
print("p-value:", kpss_result[1])

# EXPORT DATA
df2.reset_index().to_csv("gold_cleaned.csv", index=False)

df2["Month"] = df2.index.month
df2["Quarter"] = df2.index.quarter
df2["DayOfWeek"] = df2.index.day_name()
df2.reset_index().to_csv("gold_with_season.csv", index=False)