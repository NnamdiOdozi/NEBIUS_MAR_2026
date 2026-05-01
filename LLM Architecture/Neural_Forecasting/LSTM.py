"""
Temperature Forecasting Comparison: LSTM vs N-BEATS vs Toto
Predicts next-day MaxTemp for Sydney using weatherAUS.csv

Methodology notes:
- All evaluation is in raw °C — MAE/RMSE are "off by X degrees"
- All three models do rolling one-step-ahead prediction with ground-truth context
- LSTM: trained on this dataset, uses StandardScaler internally, predictions inverse-transformed
- N-BEATS: trained on this dataset, fed raw °C (has its own internal TemporalNorm —
  pre-scaling would double-normalize and hurt performance)
- Toto: zero-shot foundation model — NOT trained on this data. Fed raw °C with full
  available context (not truncated to SEQ_LEN) since it was designed for long sequences.
  Per-step loop is slow (~639 calls) but necessary for a fair rolling comparison.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

SEQ_LEN = 30
LOCATION = "Sydney"
DATA_PATH = "weatherAUS.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model selection (1=run, 0=skip) ─────────────────────────────────────────
RUN_MODELS = {
    "Naive":    0,
    "LSTM":     0,
    "LightGBM": 1,
    "N-BEATS":  0,
    "Toto":     0,
}

# ── Training hyperparameters ─────────────────────────────────────────────────
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_LR = 1e-3
LSTM_EPOCHS = 300

LGBM_N_ESTIMATORS = 300
LGBM_LEARNING_RATE = 0.05
LGBM_NUM_LEAVES = 31

NBEATS_MAX_STEPS = 300
NBEATS_STACK_TYPES = ["identity", "identity"]

TOTO_MODEL_ID = "Datadog/Toto-Open-Base-1.0"


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data(path, location):
    """Returns raw and scaled arrays plus the scaler for inverse-transforming LSTM output."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df[df["Location"] == location].sort_values("Date").dropna(subset=["MaxTemp"])
    values = df["MaxTemp"].values.reshape(-1, 1)

    split = int(len(values) * 0.8)
    train_raw, val_raw = values[:split].flatten(), values[split:].flatten()

    scaler = StandardScaler().fit(values[:split])
    train_scaled = scaler.transform(values[:split]).flatten()
    val_scaled   = scaler.transform(values[split:]).flatten()
    return train_raw, val_raw, train_scaled, val_scaled, scaler


def make_sequences(series, seq_len):
    X = np.array([series[i : i + seq_len] for i in range(len(series) - seq_len)])
    y = np.array([series[i + seq_len]      for i in range(len(series) - seq_len)])
    return X, y


# ── LSTM ──────────────────────────────────────────────────────────────────────
# Trained on this dataset. Uses StandardScaler; predictions inverse-transformed to °C.

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm   = nn.LSTM(input_size=1, hidden_size=LSTM_HIDDEN_SIZE,
                              num_layers=LSTM_NUM_LAYERS, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(LSTM_HIDDEN_SIZE, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.linear(out).squeeze(-1)


def train_lstm(X_tr, y_tr, X_val, y_val, scaler):
    model     = LSTMModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
    loss_fn   = nn.MSELoss()

    X_t = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    y_t = torch.tensor(y_tr, dtype=torch.float32).to(DEVICE)
    X_v = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    y_v = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    for epoch in range(1, LSTM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        train_loss = loss_fn(model(X_t), y_t)
        train_loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = loss_fn(model(X_v), y_v)
            print(f"  LSTM epoch {epoch}/{LSTM_EPOCHS}  train_loss={train_loss.item():.4f}  test_loss={test_loss.item():.4f}")

    param_count = sum(p.numel() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_v).cpu().numpy()
    return scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten(), param_count


# ── LightGBM ─────────────────────────────────────────────────────────────────
# Trained on this dataset. Uses same scaled lag-window features as LSTM.
# Predictions inverse-transformed to °C.

def run_lightgbm(X_tr, y_tr, X_val, scaler):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=LGBM_N_ESTIMATORS,
        learning_rate=LGBM_LEARNING_RATE,
        num_leaves=LGBM_NUM_LEAVES,
        verbose=-1,
    )
    model.fit(X_tr, y_tr)

    preds_scaled = model.predict(X_val)
    param_count = model.booster_.num_trees() * LGBM_NUM_LEAVES
    return scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten(), param_count


# ── N-BEATS ───────────────────────────────────────────────────────────────────
# Trained on this dataset. Fed raw °C — N-BEATS applies its own internal
# TemporalNorm, so pre-scaling would double-normalize.

def run_nbeats(train_raw, val_raw):
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATS

    n_val = len(val_raw) - SEQ_LEN

    full = np.concatenate([train_raw, val_raw])
    dates = pd.date_range("2007-01-01", periods=len(full), freq="D")

    full_df = pd.DataFrame({
        "unique_id": "Sydney",
        "ds":        dates,
        "y":         full,
    })

    nf = NeuralForecast(
        models=[NBEATS(h=1, input_size=SEQ_LEN, max_steps=NBEATS_MAX_STEPS, stack_types=NBEATS_STACK_TYPES)],
        freq="D",
    )

    # Rolling 1-step predictions: fits once, predicts n_val windows
    cv_df = nf.cross_validation(full_df, step_size=1, n_windows=n_val)
    param_count = sum(p.numel() for p in nf.models[0].parameters())
    return cv_df["NBEATS"].values, param_count


# ── Toto ──────────────────────────────────────────────────────────────────────
# Zero-shot foundation model — NOT trained on this data.
# Fed raw °C with full available history as context (not truncated to SEQ_LEN).
# Per-step loop is slow but gives a fair rolling one-step comparison.

def run_toto(train_raw, val_raw):
    from toto.model.toto import Toto
    from toto.inference.forecaster import TotoForecaster
    from toto.data.util.dataset import MaskedTimeseries

    n_val = len(val_raw) - SEQ_LEN

    toto = Toto.from_pretrained(TOTO_MODEL_ID)
    toto.model = toto.model.to(DEVICE)
    param_count = sum(p.numel() for p in toto.model.parameters())
    forecaster = TotoForecaster(toto.model)

    preds = []
    for i in range(n_val):
        # Full available context: all training data + ground truth up to this point
        ctx = train_raw.tolist() + val_raw[:SEQ_LEN + i].tolist()
        T = len(ctx)

        series_t = torch.tensor(ctx, dtype=torch.float32).view(1, 1, T).to(DEVICE)
        mask_t   = torch.ones(1, 1, T, dtype=torch.bool).to(DEVICE)
        id_mask  = torch.zeros(1, 1, 1, dtype=torch.long).to(DEVICE)

        inputs = MaskedTimeseries(
            series=series_t,
            padding_mask=mask_t,
            id_mask=id_mask,
            timestamp_seconds=torch.zeros(1, 1, T, dtype=torch.long).to(DEVICE),
            time_interval_seconds=torch.tensor([[86400]], dtype=torch.long).to(DEVICE),
        )
        out = forecaster.forecast(inputs, prediction_length=1, num_samples=100)
        preds.append(float(out.samples.mean().cpu()))

        if (i + 1) % 100 == 0:
            print(f"  Toto step {i+1}/{n_val}")

    return np.array(preds), param_count


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred):
    """MAE and RMSE in °C"""
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"MAE": mae, "RMSE": rmse}


def save_results(results, train_n, val_n):
    """Write timestamped results to results/ as JSON."""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    record = {
        "timestamp": datetime.now().isoformat(),
        "device": str(DEVICE),
        "location": LOCATION,
        "seq_len": SEQ_LEN,
        "train_size": train_n,
        "val_size": val_n,
        "models": results,
    }

    path = results_dir / f"run_{ts}.json"
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nResults saved to {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print(f"Loading data for {LOCATION}...")
    train_raw, val_raw, train_scaled, val_scaled, scaler = load_data(DATA_PATH, LOCATION)
    print(f"  train={len(train_raw)}  val={len(val_raw)}")

    # Ground truth in °C — the values all three models predict
    y_true_celsius = val_raw[SEQ_LEN:]

    # LSTM uses scaled sequences
    X_tr, y_tr   = make_sequences(train_scaled, SEQ_LEN)
    X_val, y_val = make_sequences(val_scaled,   SEQ_LEN)

    results = {}

    if RUN_MODELS["Naive"]:
        print("\n--- Naive (yesterday's temp) ---")
        results["Naive"] = {
            **evaluate(y_true_celsius, val_raw[SEQ_LEN - 1 : -1]),
            "config": {"param_count": 0},
        }

    if RUN_MODELS["LSTM"]:
        print("\n--- LSTM (trained, scaled input) ---")
        lstm_preds, lstm_params = train_lstm(X_tr, y_tr, X_val, y_val, scaler)
        results["LSTM"] = {
            **evaluate(y_true_celsius, lstm_preds),
            "config": {"hidden_size": LSTM_HIDDEN_SIZE, "num_layers": LSTM_NUM_LAYERS,
                        "epochs": LSTM_EPOCHS, "lr": LSTM_LR, "batch_mode": "full",
                        "param_count": lstm_params},
        }

    if RUN_MODELS["LightGBM"]:
        print("\n--- LightGBM (trained, scaled input) ---")
        lgbm_preds, lgbm_params = run_lightgbm(X_tr, y_tr, X_val, scaler)
        results["LightGBM"] = {
            **evaluate(y_true_celsius, lgbm_preds),
            "config": {"n_estimators": LGBM_N_ESTIMATORS, "learning_rate": LGBM_LEARNING_RATE,
                        "num_leaves": LGBM_NUM_LEAVES, "param_count": lgbm_params},
        }

    if RUN_MODELS["N-BEATS"]:
        print("\n--- N-BEATS (trained, raw input) ---")
        nbeats_preds, nbeats_params = run_nbeats(train_raw, val_raw)
        results["N-BEATS"] = {
            **evaluate(y_true_celsius, nbeats_preds),
            "config": {"max_steps": NBEATS_MAX_STEPS, "stack_types": NBEATS_STACK_TYPES,
                        "param_count": nbeats_params},
        }

    if RUN_MODELS["Toto"]:
        print("\n--- Toto (zero-shot, raw input) ---")
        toto_preds, toto_params = run_toto(train_raw, val_raw)
        results["Toto"] = {
            **evaluate(y_true_celsius, toto_preds),
            "config": {"model_id": TOTO_MODEL_ID, "zero_shot": True,
                        "param_count": toto_params},
        }

    # All metrics in °C
    print("\n" + "=" * 42)
    print(f"{'Model':<12} {'MAE (°C)':>13} {'RMSE (°C)':>13}")
    print("-" * 42)
    for model, data in results.items():
        print(f"{model:<12} {data['MAE']:>13.2f} {data['RMSE']:>13.2f}")
    print("=" * 42)

    save_results(results, len(train_raw), len(val_raw))


if __name__ == "__main__":
    main()
