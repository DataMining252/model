# 1. Tạo target cho 7 ngày tới
```bash
df['Target_1'] = df['Close'].shift(-1)
df['Target_2'] = df['Close'].shift(-2)
df['Target_3'] = df['Close'].shift(-3)
df['Target_4'] = df['Close'].shift(-4)
df['Target_5'] = df['Close'].shift(-5)
df['Target_6'] = df['Close'].shift(-6)
df['Target_7'] = df['Close'].shift(-7)
```
Model học 7 giá trị trong tương lai

# 2. Tạo lag feature
Vì Random Forest không hiểu time series nên cần lag.
```bash
df['Close_lag1'] = df['Close'].shift(1)
df['Close_lag2'] = df['Close'].shift(2)
df['Close_lag3'] = df['Close'].shift(3)
df['Close_lag7'] = df['Close'].shift(7)
df['Close_lag14'] = df['Close'].shift(14)
```