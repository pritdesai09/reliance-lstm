from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os

app = Flask(__name__)
CORS(app)

# ── globals ───────────────────────────────────────────────────────────────────
model        = None
scaler       = None
close_scaler = None
scaled_df    = None
real_metrics = {}
real_chart   = {}
SEQ_LENGTH   = 60

BASE_DIR    = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(BASE_DIR, "saved_model", "lstm_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "saved_model", "scalers.pkl")
META_PATH   = os.path.join(BASE_DIR, "saved_model", "meta.pkl")
CSV_PATH    = os.path.join(BASE_DIR, "data", "datanewfinal.csv")

# ── helpers ───────────────────────────────────────────────────────────────────
def build_scaled_df(csv_path):
    global scaler, close_scaler
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    feature_cols = ["Close", "High", "Low", "Open", "Volume"]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])
    sdf = pd.DataFrame(scaled_data, columns=feature_cols, index=df.index)
    close_scaler = MinMaxScaler()
    close_scaler.min_   = np.array([scaler.min_[0]])
    close_scaler.scale_ = np.array([scaler.scale_[0]])
    return sdf

def compute_metrics_and_chart(X_test, y_test, test_dates):
    """Run model on test set, compute real metrics and chart data."""
    global real_metrics, real_chart

    pred_scaled = model.predict(X_test, verbose=0)
    pred_prices = close_scaler.inverse_transform(pred_scaled).flatten()
    actual_prices = close_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    rmse = float(np.sqrt(mean_squared_error(actual_prices, pred_prices)))
    mae  = float(mean_absolute_error(actual_prices, pred_prices))
    r2   = float(r2_score(actual_prices, pred_prices))
    mape = float(np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100)

    real_metrics = {"rmse": round(rmse, 2), "mae": round(mae, 2),
                    "r2": round(r2, 4), "mape": round(mape, 2)}

    # send last 60 test points to keep chart readable
    n = min(60, len(actual_prices))
    real_chart = {
        "labels":    [d.strftime("%d %b") for d in test_dates[-n:]],
        "actual":    [round(float(v), 2) for v in actual_prices[-n:]],
        "predicted": [round(float(v), 2) for v in pred_prices[-n:]]
    }
    print(f"[INFO] Real metrics — RMSE: {rmse:.2f}  MAE: {mae:.2f}  R²: {r2:.4f}  MAPE: {mape:.2f}%")

def save_artifacts():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump({"scaler": scaler, "close_scaler": close_scaler}, f)
    with open(META_PATH, "wb") as f:
        pickle.dump({"metrics": real_metrics, "chart": real_chart}, f)
    print("[INFO] Model, scalers and metrics saved.")

def load_artifacts():
    global model, scaler, close_scaler, real_metrics, real_chart
    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        d = pickle.load(f)
    scaler       = d["scaler"]
    close_scaler = d["close_scaler"]
    with open(META_PATH, "rb") as f:
        m = pickle.load(f)
    real_metrics = m["metrics"]
    real_chart   = m["chart"]
    print(f"[INFO] Loaded model. Metrics — RMSE: {real_metrics['rmse']}  R²: {real_metrics['r2']}")

# ── load & train ──────────────────────────────────────────────────────────────
def load_and_train():
    global model, scaled_df
    if not os.path.exists(CSV_PATH):
        print(f"[WARN] Dataset not found at {CSV_PATH}.")
        return False

    scaled_df = build_scaled_df(CSV_PATH)

    # build sequences
    X, y = [], []
    for i in range(len(scaled_df) - SEQ_LENGTH):
        X.append(scaled_df.iloc[i:i + SEQ_LENGTH].values)
        y.append(scaled_df.iloc[i + SEQ_LENGTH]["Close"])
    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # dates for test set (for chart labels)
    test_start_idx = len(X_train) + SEQ_LENGTH
    test_dates = scaled_df.index[test_start_idx: test_start_idx + len(X_test)]

    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(META_PATH):
        print("[INFO] Saved model found — loading from disk (skipping training)...")
        load_artifacts()
        close_scaler.min_   = np.array([scaler.min_[0]])
        close_scaler.scale_ = np.array([scaler.scale_[0]])
        print("[INFO] Ready!")
        return True

    print("[INFO] No saved model — training from scratch...")
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=100, batch_size=32, callbacks=[early_stop], verbose=1)

    print("[INFO] Computing real metrics on test set...")
    compute_metrics_and_chart(X_test, y_test, test_dates)
    save_artifacts()
    return True

# ── prediction ────────────────────────────────────────────────────────────────
def predict_price_on(target_date_str):
    target_date = pd.to_datetime(target_date_str)
    last_date   = scaled_df.index[-1]
    if target_date > last_date:
        days_ahead = (target_date - last_date).days
        current_sequence = np.copy(scaled_df[-SEQ_LENGTH:].values)
        future_predictions = []
        for _ in range(days_ahead):
            input_seq   = np.expand_dims(current_sequence, axis=0)
            pred_scaled = model.predict(input_seq, verbose=0)
            next_row    = current_sequence[-1].copy()
            next_row[0] = pred_scaled[0][0]
            current_sequence = np.vstack((current_sequence[1:], next_row))
            future_predictions.append(close_scaler.inverse_transform(pred_scaled)[0][0])
        return float(future_predictions[-1])
    else:
        if target_date not in scaled_df.index:
            available = scaled_df.index[scaled_df.index < target_date]
            if len(available) == 0:
                return None
            target_date = available[-1]
        idx = scaled_df.index.get_loc(target_date)
        if idx < SEQ_LENGTH:
            return None
        seq_data = np.expand_dims(scaled_df.iloc[idx - SEQ_LENGTH:idx].values, axis=0)
        pred_scaled = model.predict(seq_data, verbose=0)
        return float(close_scaler.inverse_transform(pred_scaled)[0][0])

# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 503
    data = request.json
    date_str = data.get("date")
    if not date_str:
        return jsonify({"error": "Missing 'date' field"}), 400
    try:
        price = predict_price_on(date_str)
        if price is None:
            return jsonify({"error": "Not enough data for this date"}), 400
        return jsonify({"predicted_price": round(price, 2), "date": date_str})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None,
                    "data_points": len(scaled_df) if scaled_df is not None else 0})

@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify({**real_metrics,
                    "architecture": "LSTM(64) → LSTM(32) → Dense(1)",
                    "seq_length": SEQ_LENGTH,
                    "training_period": "Mar 2021 – Feb 2025"})

@app.route("/chart-data", methods=["GET"])
def chart_data():
    return jsonify(real_chart)

# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[INFO] Starting up...")
    load_and_train()
    print("[INFO] Flask server running at http://localhost:5000")
    app.run(debug=False, port=5000)