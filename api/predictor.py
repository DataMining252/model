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
    map_location="cpu",
    weights_only=False
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
    def __init__(self, input_size):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.fc = torch.nn.Linear(64, HORIZON)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


lstm_model = LSTMModel(len(lstm_features))
lstm_model.load_state_dict(lstm_bundle["model_state_dict"])
lstm_model.eval()


# ================= FEATURE ENGINEERING =================
def build_features(df: pd.DataFrame):
    df = df.copy()

    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['high-low'] = df['high'] - df['low']
    df['close-open'] = df['close'] - df['open']

    df['ma_7'] = df['close'].rolling(7).mean()
    df['ma_14'] = df['close'].rolling(14).mean()
    df['std_7'] = df['close'].rolling(7).std()

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

    # chỉ lấy đúng feature model train
    lstm_input = feat_df[lstm_features].iloc[-WINDOW_SIZE:].copy()

    # giữ price riêng (QUAN TRỌNG)
    price_series = hist_df['close'].iloc[-len(feat_df):].reset_index(drop=True)

    pred_lstm_list = []

    for _ in range(n_forecast_days):

        # ================= SCALE =================
        if scaler_X is not None:
            X_scaled = scaler_X.transform(lstm_input.values)
        else:
            X_scaled = minmax_scale_local(lstm_input.values)

        X_seq = torch.FloatTensor(X_scaled).unsqueeze(0)

        # ================= PREDICT =================
        with torch.no_grad():
            lstm_out = lstm_model(X_seq).cpu().numpy()[0]

        # multi-horizon output -> lấy mean ổn định
        pred_log_return = float(np.mean(lstm_out))

        # ================= INVERSE SCALE =================
        if scaler_y is not None:
            pred_log_return = scaler_y.inverse_transform([[pred_log_return]])[0, 0]

        # ================= CONVERT TO PRICE =================
        last_price = price_series.iloc[-1]
        pred_price = last_price * np.exp(pred_log_return)

        pred_lstm_list.append(pred_price)

        # ================= UPDATE PRICE SERIES =================
        price_series = pd.concat(
            [price_series, pd.Series([pred_price])],
            ignore_index=True
        )

        # ================= REBUILD FEATURES (IMPORTANT FIX) =================
        temp_df = hist_df.copy()
        temp_df = pd.concat(
            [temp_df, pd.DataFrame([{"close": pred_price}])],
            ignore_index=True
        )

        feat_df = build_features(temp_df)
        feat_df.columns = feat_df.columns.str.lower()

        lstm_input = feat_df[lstm_features].iloc[-WINDOW_SIZE:].copy()

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