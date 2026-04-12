import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path

# ======================
# CONFIG
# ======================
WINDOW_SIZE = 60
HORIZON = 7
EPOCHS = 100
LR = 0.001
PATIENCE = 10

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "raw" / "final_dataset.csv"
SAVE_DIR = BASE_DIR / "results"
SAVE_DIR.mkdir(exist_ok=True)

MODEL_PATH = BASE_DIR / "models" / "lstm_price_model_final.pth"
MODEL_PATH.parent.mkdir(exist_ok=True)

# ======================
# LOAD DATA
# ======================
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# ======================
# FEATURE ENGINEERING
# ======================
# ✅ LOG RETURN (QUAN TRỌNG)
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

df['High-Low'] = df['High'] - df['Low']
df['Close-Open'] = df['Close'] - df['Open']

df['ma_7'] = df['Close'].rolling(7).mean()
df['ma_14'] = df['Close'].rolling(14).mean()
df['std_7'] = df['Close'].rolling(7).std()

df = df.dropna()

features = [
    'log_return','High-Low','Close-Open',
    'ma_7','ma_14','std_7',
    'DXY','SP500','OIL'
]

X_data = df[features].values
y_data = df['log_return'].values.reshape(-1,1)

# ======================
# SPLIT
# ======================
n = len(X_data)
train_end = int(n*0.7)
val_end = int(n*0.8)

# ======================
# SCALE (NO LEAKAGE)
# ======================
scaler_X = MinMaxScaler()
scaler_y = StandardScaler()  # ✅ FIX

scaler_X.fit(X_data[:train_end])
scaler_y.fit(y_data[:train_end])

X_scaled = scaler_X.transform(X_data)
y_scaled = scaler_y.transform(y_data)

# ======================
# CREATE SEQUENCE
# ======================
def create_sequences(X, y, window, horizon):
    X_seq, y_seq = [], []
    for i in range(len(X)-window-horizon):
        X_seq.append(X[i:i+window])
        y_seq.append(y[i+window:i+window+horizon, 0])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, WINDOW_SIZE, HORIZON)

# split
n_seq = len(X_seq)
train_end = int(n_seq*0.7)
val_end = int(n_seq*0.8)

X_train, y_train = X_seq[:train_end], y_seq[:train_end]
X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
X_test, y_test = X_seq[val_end:], y_seq[val_end:]

# tensor
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# ======================
# MODEL
# ======================
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(64, HORIZON)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

model = LSTMModel(len(features))

# FIX LOSS
criterion = nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ======================
# TRAIN + EARLY STOPPING
# ======================
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val)

    print(f"Epoch {epoch+1} | Train: {loss.item():.6f} | Val: {val_loss.item():.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_features": features,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "window_size": WINDOW_SIZE,
            "horizon": HORIZON
        }, MODEL_PATH)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping!")
            break

checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
features = checkpoint["input_features"]
scaler_X = checkpoint["scaler_X"]
scaler_y = checkpoint["scaler_y"]
WINDOW_SIZE = checkpoint["window_size"]
HORIZON = checkpoint["horizon"]

model.eval()

# ======================
# PREDICT
# ======================
with torch.no_grad():
    y_pred_scaled = model(X_test).detach().cpu().numpy()
    y_test_np = y_test.detach().cpu().numpy()

# ======================
# INVERSE SCALE
# ======================
y_pred_log = scaler_y.inverse_transform(y_pred_scaled)
y_true_log = scaler_y.inverse_transform(y_test_np)

# ======================
# RECONSTRUCT PRICE (LOG RETURN)
# ======================
start_idx = val_end + WINDOW_SIZE
base_prices = df['Close'].values[start_idx:start_idx+len(y_pred_log)]

def reconstruct_price(base, log_returns):
    prices = []
    for i in range(len(log_returns)):
        p = base[i]
        seq = []
        for r in log_returns[i]:
            p = p * np.exp(r)  # ✅ FIX
            seq.append(p)
        prices.append(seq)
    return np.array(prices)

y_pred_price = reconstruct_price(base_prices, y_pred_log)
y_true_price = reconstruct_price(base_prices, y_true_log)

# ======================
# EVALUATE
# ======================
print("\n===== LSTM METRICS =====")
for i in range(HORIZON):
    mae = mean_absolute_error(y_true_price[:,i], y_pred_price[:,i])
    rmse = np.sqrt(mean_squared_error(y_true_price[:,i], y_pred_price[:,i]))
    r2 = r2_score(y_true_price[:,i], y_pred_price[:,i])
    print(f"Day+{i+1}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")

# ======================
# CHECK BIAS
# ======================
bias = np.mean(y_pred_price - y_true_price)
print("\nBias:", bias)

# ======================
# PLOT
# ======================
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 2, figsize=(15, 12))
axes = axes.flatten()

for i in range(HORIZON):
    actual = y_true_price[:, i]
    pred = y_pred_price[:, i]

    # ======================
    # LINE PLOT (Left side)
    # ======================
    ax = axes[i]
    ax.plot(actual[-100:], label="Actual")
    ax.plot(pred[-100:], linestyle="--", label="Predicted")

    ax.set_title(f"Day +{i+1} (Line)")
    ax.grid(True)

# remove unused subplot (if HORIZON < 8)
if HORIZON < 8:
    fig.delaxes(axes[7])

# ======================
# SCATTER FULL HORIZON
# ======================
fig2, axes2 = plt.subplots(4, 2, figsize=(15, 12))
axes2 = axes2.flatten()

for i in range(HORIZON):
    ax = axes2[i]

    actual = y_true_price[:, i]
    pred = y_pred_price[:, i]

    ax.scatter(actual, pred, alpha=0.4)

    # đường y = x
    min_v = min(actual.min(), pred.min())
    max_v = max(actual.max(), pred.max())
    ax.plot([min_v, max_v], [min_v, max_v], 'r--')

    ax.set_title(f"Day +{i+1} (Scatter)")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid(True)

# remove unused axis nếu HORIZON < 8
for j in range(HORIZON, 8):
    fig2.delaxes(axes2[j])

plt.tight_layout()

# ======================
# SAVE
# ======================
SAVE_DIR.mkdir(parents=True, exist_ok=True)

fig.savefig(SAVE_DIR / "lstm_line_plot.png", dpi=300, bbox_inches="tight")
fig2.savefig(SAVE_DIR / "lstm_scatter_plot.png", dpi=300, bbox_inches="tight")

plt.close(fig)
plt.close(fig2)