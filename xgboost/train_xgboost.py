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

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "raw" / "final_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "lstm_price_model_final.pth"
SAVE_DIR = BASE_DIR / "results"
SAVE_DIR.mkdir(exist_ok=True)
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
    'Open','High','Low','Close','DXY','SP500','OIL','INTEREST_RATE','CPI',
    'High-Low','Close-Open','rolling_mean_7','rolling_std_7'
]

# Xác định vị trí các features
print("Features order:")
for idx, f in enumerate(features):
    print(f"  {idx}: {f}")

close_idx = features.index('Close')
print(f"\nClose index: {close_idx}")

X_data = df[features].values
y_data = df['Close'].values.reshape(-1,1)

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
def create_sequences_recursive(X, y, window):
    X_seq, y_seq = [], []
    for i in range(len(X) - window):
        X_seq.append(X[i:i+window])
        y_seq.append(y[i+window, 0])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences_recursive(X_scaled, y_scaled, WINDOW_SIZE)

# ======================
# SPLIT TRAIN/TEST
# ======================
split = int(len(X_seq) * 0.8)
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
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=64, 
                           num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
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
    loss = criterion(pred_train.squeeze(), y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.6f}")

# ======================
# SAVE MODEL
# ======================
torch.save({
    'model_state_dict': model.state_dict(),
    'input_features': features,
    'window_size': WINDOW_SIZE,
    'horizon': HORIZON
}, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# ======================
# RECURSIVE FORECAST (CẢI TIẾN)
# ======================
def recursive_forecast_v2(model, initial_sequence, steps, scaler_X, scaler_y, feature_indices):
    """
    Dự báo recursive với cập nhật đầy đủ features
    """
    model.eval()
    predictions_scaled = []
    current_seq = initial_sequence.clone()
    
    with torch.no_grad():
        for step in range(steps):
            # Dự báo Close scaled
            pred_scaled = model(current_seq.unsqueeze(0))
            pred_close_scaled = pred_scaled[0, 0].item()
            predictions_scaled.append(pred_close_scaled)
            
            # Tạo features cho ngày mới (dựa trên ngày cuối cùng)
            last_features = current_seq[-1].clone()
            
            # Cập nhật Close
            last_features[feature_indices['close']] = pred_close_scaled
            
            # Cập nhật Open = Close của ngày trước (giả định)
            if step == 0:
                last_features[feature_indices['open']] = current_seq[-1, feature_indices['close']]
            else:
                last_features[feature_indices['open']] = last_features[feature_indices['close']]
            
            # Cập nhật High = Close * 1.002 (ước lượng)
            last_features[feature_indices['high']] = last_features[feature_indices['close']] * 1.002
            
            # Cập nhật Low = Close * 0.998 (ước lượng)
            last_features[feature_indices['low']] = last_features[feature_indices['close']] * 0.998
            
            # Cập nhật Close-Open
            last_features[feature_indices['close-open']] = last_features[feature_indices['close']] - last_features[feature_indices['open']]
            
            # Cập nhật High-Low
            last_features[feature_indices['high-low']] = last_features[feature_indices['high']] - last_features[feature_indices['low']]
            
            # Cập nhật rolling_mean_7 (trung bình 7 ngày gần nhất)
            recent_closes = torch.cat([current_seq[-6:, feature_indices['close']], torch.tensor([pred_close_scaled])])
            last_features[feature_indices['rolling_mean_7']] = recent_closes.mean()
            
            # Cập nhật rolling_std_7
            last_features[feature_indices['rolling_std_7']] = recent_closes.std()
            
            # Thêm bước mới
            current_seq = torch.cat([current_seq[1:], last_features.unsqueeze(0)], dim=0)
    
    # Inverse transform
    predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
    return scaler_y.inverse_transform(predictions_scaled).flatten()

# ======================
# EVALUATE
# ======================
feature_indices = {
    'open': features.index('Open'),
    'high': features.index('High'),
    'low': features.index('Low'),
    'close': features.index('Close'),
    'close-open': features.index('Close-Open'),
    'high-low': features.index('High-Low'),
    'rolling_mean_7': features.index('rolling_mean_7'),
    'rolling_std_7': features.index('rolling_std_7')
}

model.eval()
y_true_all = []
y_pred_all = []

print("Đang dự báo recursive...")
with torch.no_grad():
    for i in range(min(len(X_test) - HORIZON, 200)):  # Giới hạn 200 mẫu để chạy nhanh
        if i % 50 == 0:
            print(f"  Đang xử lý mẫu {i}/{min(len(X_test) - HORIZON, 200)}")
        
        initial_seq = X_test[i]
        true_values = [y_test[i + h].item() for h in range(HORIZON)]
        
        # Inverse scale true values
        true_values_scaled = np.array(true_values).reshape(-1, 1)
        true_values_original = scaler_y.inverse_transform(true_values_scaled).flatten()
        
        pred_values = recursive_forecast_v2(model, initial_seq, HORIZON, scaler_X, scaler_y, feature_indices)
        
        y_true_all.append(true_values_original)
        y_pred_all.append(pred_values)

y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

# ======================
# PRINT METRICS
# ======================
print("\n" + "="*70)
print("                     LSTM RECURSIVE FORECASTING RESULTS")
print("="*70)
print(f"{'Horizon':<12} {'MAE':<12} {'RMSE':<12} {'R²':<12}")
print("-"*70)

metrics = []
for i in range(HORIZON):
    actual = y_true_all[:, i]
    pred = y_pred_all[:, i]
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)
    metrics.append((mae, rmse, r2))
    print(f"Day +{i+1:<7} {mae:<12.2f} {rmse:<12.2f} {r2:<12.4f}")

print("-"*70)
print(f"{'Trung bình':<12} {np.mean([m[0] for m in metrics]):<12.2f} {np.mean([m[1] for m in metrics]):<12.2f} {np.mean([m[2] for m in metrics]):<12.4f}")
print("="*70)

# ======================
# PLOTS
# ======================
fig, axes = plt.subplots(4, 2, figsize=(15, 12))
axes = axes.flatten()

for i in range(min(HORIZON, 7)):
    ax = axes[i]
    actual = y_true_all[-100:, i]
    pred = y_pred_all[-100:, i]
    
    ax.plot(actual, label="Actual", linewidth=1.5)
    ax.plot(pred, linestyle="--", label="Predicted", linewidth=1.5)
    ax.set_title(f"Day +{i+1} (MAE={metrics[i][0]:.2f})")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Gold Price")
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.delaxes(axes[7])
plt.tight_layout()
plt.savefig(SAVE_DIR / "lstm_line_plot.png", dpi=150)
plt.close()

# Scatter plot
fig, axes = plt.subplots(4, 2, figsize=(15, 12))
axes = axes.flatten()

for i in range(min(HORIZON, 7)):
    ax = axes[i]
    actual = y_true_all[-100:, i]
    pred = y_pred_all[-100:, i]
    
    ax.scatter(actual, pred, alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    min_val = min(actual.min(), pred.min())
    max_val = max(actual.max(), pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label="y=x")
    ax.set_title(f"Day +{i+1}")
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.delaxes(axes[7])
plt.tight_layout()
plt.savefig(SAVE_DIR / "lstm_scatter_plot.png", dpi=150)
plt.close()

print(f"\n✅ Done! Results saved to {SAVE_DIR}")