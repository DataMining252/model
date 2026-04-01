import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# ======================
# CONFIG
# ======================
WINDOW_SIZE = 60
HORIZON = 7
EPOCHS = 50
LR = 0.001

from pathlib import Path

# Thư mục chứa file .py hiện tại
BASE_DIR = Path(__file__).resolve().parent

# ../raw/final_dataset.csv
DATA_PATH = BASE_DIR.parent / "raw" / "final_dataset.csv"

# ./models/lstm_price_model_final.pth
MODEL_PATH = BASE_DIR / "models" / "lstm_price_model_final.pth"

os.makedirs("./models", exist_ok=True)

# ======================
# LOAD DATA
# ======================
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# ======================
# FEATURE ENGINEERING
# ======================
df['High-Low'] = df['High'] - df['Low']
df['Close-Open'] = df['Close'] - df['Open']
df['rolling_mean_7'] = df['Close'].rolling(7).mean()
df['rolling_std_7'] = df['Close'].rolling(7).std()
df = df.dropna()

features = [
    'Open','High','Low','Close','Volume','DXY','SP500','OIL','INTEREST_RATE','CPI',
    'High-Low','Close-Open','rolling_mean_7','rolling_std_7'
]

X_data = df[features].values
y_data = df['Close'].values.reshape(-1,1)  # dự đoán giá trực tiếp

# ======================
# SCALE
# ======================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X_data)
y_scaled = scaler_y.fit_transform(y_data)

# ======================
# CREATE SEQUENCES
# ======================
def create_sequences(X, y, window, horizon):
    X_seq, y_seq = [], []
    for i in range(len(X)-window-horizon):
        X_seq.append(X[i:i+window])
        y_seq.append(y[i+window:i+window+horizon, 0])  # flatten last dim
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, WINDOW_SIZE, HORIZON)

# ======================
# SPLIT TRAIN/TEST
# ======================
split = int(len(X_seq)*0.8)
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# ======================
# LSTM MODEL
# ======================
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=256, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(256, HORIZON)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # lấy output timestep cuối
        return self.fc(out)

model = LSTMModel(input_size=len(features))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ======================
# TRAIN
# ======================
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    pred_train = model(X_train)
    loss = criterion(pred_train, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.6f}")

# ======================
# SAVE MODEL
# ======================
MODEL_PATH = Path(MODEL_PATH)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'input_features': features,
    'window_size': WINDOW_SIZE,
    'horizon': HORIZON
}, MODEL_PATH)
print(f"LSTM model saved to {MODEL_PATH}")

# ======================
# PREDICT
# ======================
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test).numpy()
y_test_np = y_test.numpy()

# ======================
# INVERSE SCALE
# ======================
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test_np)

# ======================
# LINE PLOT
# ======================
fig, axes = plt.subplots(4, 2, figsize=(15,12))
axes = axes.flatten()
for i in range(HORIZON):
    ax = axes[i]
    actual = y_true[:,i]
    pred = y_pred[:,i]
    ax.plot(actual[-100:], label="Actual")
    ax.plot(pred[-100:], linestyle="--", label="Predicted")
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)
    ax.set_title(f"Day +{i+1} | MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.2f}")
    ax.grid(True)
fig.delaxes(axes[7])
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels)
plt.tight_layout()
plt.show()

# ======================
# SCATTER PLOT
# ======================
fig, axes = plt.subplots(4, 2, figsize=(15,12))
axes = axes.flatten()
for i in range(HORIZON):
    ax = axes[i]
    actual = y_true[:,i]
    pred = y_pred[:,i]
    ax.scatter(actual[-100:], pred[-100:], alpha=0.7, s=20)
    ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')  # y=x line
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)
    ax.set_title(f"Day +{i+1} | MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.2f}")
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.grid(True)
fig.delaxes(axes[7])
plt.tight_layout()
plt.show()
