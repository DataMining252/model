import os
import pandas as pd
import torch
import joblib
import numpy as np

# ================= LOAD MODELS =================
BASE_DIR = os.path.dirname(__file__)

# ---------- XGBoost ----------
xgb_bundle = joblib.load(
    os.path.join(BASE_DIR, "..", "xgboost", "models", "xgb_model.pkl")
)
xgb_models = xgb_bundle["models"]
xgb_features = xgb_bundle["features"]

# ---------- LSTM ----------
lstm_bundle = torch.load(
    os.path.join(BASE_DIR, "..", "lstm", "models", "lstm_price_model_final.pth"),
    map_location="cpu"
)

lstm_features = lstm_bundle["input_features"]
lstm_features = [f.lower() for f in lstm_features]
WINDOW_SIZE = lstm_bundle["window_size"]
HORIZON = lstm_bundle["horizon"]

# scaler có thể không tồn tại
scaler_X = lstm_bundle.get("scaler_X", None)
scaler_y = lstm_bundle.get("scaler_y", None)


# ================= MODEL =================
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, horizon):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.fc = torch.nn.Linear(256, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


lstm_model = LSTMModel(len(lstm_features), HORIZON)
lstm_model.load_state_dict(lstm_bundle["model_state_dict"])
lstm_model.eval()


# ================= FEATURE ENGINEERING =================
def build_features(df: pd.DataFrame):
    df = df.copy()

    df['high-low'] = df['high'] - df['low']
    df['close-open'] = df['close'] - df['open']
    df['rolling_mean_7'] = df['close'].rolling(7).mean()
    df['rolling_std_7'] = df['close'].rolling(7).std()

    return df.dropna()


def build_xgb_features(df: pd.DataFrame):
    df = df.copy()

    # ===== LAG =====
    for i in range(1, 15):
        df[f'lag_{i}'] = df['close'].shift(i)

    # ===== ROLLING =====
    df['ma_7'] = df['close'].rolling(7).mean()
    df['std_7'] = df['close'].rolling(7).std()

    # ===== RETURN =====
    df['return_1'] = df['close'].pct_change(1)
    df['return_7'] = df['close'].pct_change(7)

    # ===== MOMENTUM =====
    df['momentum_7'] = df['close'] - df['close'].shift(7)

    return df.dropna()


# ================= FALLBACK SCALER =================
def minmax_scale_local(X: np.ndarray):
    min_val = X.min(axis=0)
    max_val = X.max(axis=0)
    return (X - min_val) / (max_val - min_val + 1e-8)


# ================= SERVICE =================
def predict_forecast(hist_df: pd.DataFrame, forecast_dates, n_forecast_days=7):
    if hist_df.empty:
        raise ValueError("No historical data")

    # ================= XGBoost =================
    xgb_preds = []
    temp_df = build_xgb_features(hist_df)

    for i in range(1, n_forecast_days + 1):
        X = temp_df[xgb_features].iloc[-1].values.reshape(1, -1)
        pred = xgb_models[min(i, 7)].predict(X)[0]
        xgb_preds.append(float(pred))

        # update recursive
        new_row = temp_df.iloc[-1].copy()
        new_row['Close'] = pred
        temp_df = pd.concat(
            [temp_df, pd.DataFrame([new_row])],
            ignore_index=True
        )

    # ================= LSTM =================
    feat_df = build_features(hist_df)
    feat_df.columns = feat_df.columns.str.lower()

    if len(feat_df) < WINDOW_SIZE:
        raise ValueError("Not enough data for LSTM")

    # LSTM forecast tuần tự
    lstm_input = feat_df[lstm_features].iloc[-WINDOW_SIZE:].copy()
    pred_lstm_list = []

    for _ in range(n_forecast_days):
        # scale input
        if scaler_X is not None:
            X_scaled = scaler_X.transform(lstm_input.values)
        else:
            X_scaled = minmax_scale_local(lstm_input.values)

        X_seq = torch.FloatTensor(X_scaled).unsqueeze(0)

        with torch.no_grad():
            lstm_out = lstm_model(X_seq).numpy()[0, 0]

        # convert về USD
        if scaler_y is not None:
            lstm_pred = scaler_y.inverse_transform([[lstm_out]])[0, 0]
        else:
            last_price = lstm_input['close'].iloc[-1]
            lstm_out = max(min(float(lstm_out), 0.05), -0.05)
            lstm_pred = last_price * (1 + lstm_out)

        pred_lstm_list.append(lstm_pred)

        # update window cho next day
        new_row = lstm_input.iloc[-1].copy()
        new_row['close'] = lstm_pred
        lstm_input = pd.concat([lstm_input.iloc[1:], new_row.to_frame().T], ignore_index=True)

    # ================= COMBINE =================
    last_price = hist_df['close'].iloc[-1]  # giá gốc cuối cùng
    forecast = []
    for d, v, t in zip(forecast_dates, xgb_preds, pred_lstm_list):
        forecast.append({
            "date": d.isoformat(),
            "prediction_value": last_price * (1 + v),       # XGBoost USD
            "prediction_lstm": t                            # LSTM USD
        })

    return forecast