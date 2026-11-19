"""Shared pipeline for the natural gas forecasting experiments.

Every notebook in ``notebooks/`` uses the same data loading, windowing,
splitting, tuning, evaluation and forecasting code from this module, so the
six models differ only in their architecture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import keras
import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "dataset.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT / "models"
TUNING_DIR = ROOT / "tuning"

SEED = 42
LOOKBACK = 30
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # the remaining 15% is the held-out test set
MAX_TRIALS = 10
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 10
FORECAST_DAYS = 90


def set_seed(seed: int = SEED) -> None:
    """Make runs reproducible (Python, NumPy and TensorFlow RNGs + deterministic ops)."""
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


# --------------------------------------------------------------------------- data
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the daily natural-gas power generation series (GWh)."""
    data = pd.read_csv(path)
    data["date"] = pd.to_datetime(data["date"], format="%d/%m/%Y")
    return data.sort_values("date").reset_index(drop=True)


@dataclass
class Dataset:
    dates: pd.Series
    scaler: MinMaxScaler
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    last_window: np.ndarray  # most recent LOOKBACK scaled values, for forecasting

    @property
    def split_dates(self) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Target dates of the train, validation and test windows."""
        target_dates = self.dates.iloc[LOOKBACK:].reset_index(drop=True)
        n_tr, n_va = len(self.y_train), len(self.y_val)
        return (
            target_dates.iloc[:n_tr],
            target_dates.iloc[n_tr : n_tr + n_va],
            target_dates.iloc[n_tr + n_va :],
        )


def prepare_dataset(data: pd.DataFrame, lookback: int = LOOKBACK) -> Dataset:
    """Chronological 70/15/15 split into sliding windows.

    The scaler is fitted on the training period only, so no information from
    the validation or test periods leaks into preprocessing.
    """
    values = data[["GWh"]].to_numpy(dtype="float32")
    n_windows = len(values) - lookback
    n_train = int(n_windows * TRAIN_FRAC)
    n_val = int(n_windows * VAL_FRAC)

    scaler = MinMaxScaler()
    scaler.fit(values[: lookback + n_train])
    scaled = scaler.transform(values)

    x = np.stack([scaled[i - lookback : i] for i in range(lookback, len(scaled))])
    y = scaled[lookback:]

    return Dataset(
        dates=data["date"],
        scaler=scaler,
        x_train=x[:n_train],
        y_train=y[:n_train],
        x_val=x[n_train : n_train + n_val],
        y_val=y[n_train : n_train + n_val],
        x_test=x[n_train + n_val :],
        y_test=y[n_train + n_val :],
        last_window=scaled[-lookback:],
    )


# ------------------------------------------------------------------------- models
def _compile(inputs, outputs) -> keras.Model:
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def _pool_head(x, hp):
    """LayerNorm -> global average pooling -> dropout -> linear output."""
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(hp.Float("final_dropout", 0.1, 0.5, step=0.1))(x)
    return layers.Dense(1, activation="linear")(x)


def _tcn(hp, return_sequences=True):
    from tcn import TCN  # imported lazily: only the TCN models need keras-tcn

    return TCN(
        nb_filters=hp.Choice("nb_filters", [8, 16, 32, 64]),
        kernel_size=hp.Choice("kernel_size", [2, 3, 5]),
        nb_stacks=hp.Choice("nb_stacks", [1, 2]),
        dilations=[2**i for i in range(hp.Choice("n_dilations", [2, 3, 4]))],
        padding="causal",
        use_skip_connections=True,
        dropout_rate=hp.Float("dropout_rate", 0.1, 0.5, step=0.1),
        return_sequences=return_sequences,
    )


def build_lstm_vanilla(hp) -> keras.Model:
    inputs = layers.Input(shape=(LOOKBACK, 1))
    x = layers.LSTM(hp.Choice("lstm_units", [16, 32, 64, 128]))(inputs)
    return _compile(inputs, layers.Dense(1, activation="linear")(x))


def build_lstm_attention(hp) -> keras.Model:
    inputs = layers.Input(shape=(LOOKBACK, 1))
    x = layers.LSTM(hp.Choice("lstm_units", [8, 16, 32, 64]), return_sequences=True)(inputs)
    x = layers.MultiHeadAttention(
        num_heads=hp.Choice("num_heads", [2, 4]), key_dim=hp.Choice("key_dim", [4, 8])
    )(x, x)
    return _compile(inputs, _pool_head(x, hp))


def build_gru_vanilla(hp) -> keras.Model:
    inputs = layers.Input(shape=(LOOKBACK, 1))
    x = layers.GRU(hp.Choice("gru_units", [8, 16, 32]), return_sequences=True)(inputs)
    return _compile(inputs, _pool_head(x, hp))


def build_gru_attention(hp) -> keras.Model:
    inputs = layers.Input(shape=(LOOKBACK, 1))
    x = layers.GRU(hp.Choice("gru_units", [8, 16, 32]), return_sequences=True)(inputs)
    x = layers.MultiHeadAttention(
        num_heads=hp.Choice("num_heads", [2, 4]), key_dim=hp.Choice("key_dim", [8, 16])
    )(x, x)
    return _compile(inputs, _pool_head(x, hp))


def build_tcn_vanilla(hp) -> keras.Model:
    inputs = layers.Input(shape=(LOOKBACK, 1))
    x = _tcn(hp)(inputs)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(hp.Float("final_dropout", 0.1, 0.5, step=0.1))(x)
    return _compile(inputs, layers.Dense(1, activation="linear")(x))


def build_tcn_attention(hp) -> keras.Model:
    inputs = layers.Input(shape=(LOOKBACK, 1))
    x = _tcn(hp)(inputs)
    x = layers.MultiHeadAttention(num_heads=2, key_dim=hp.Choice("key_dim", [8, 16]))(x, x)
    return _compile(inputs, _pool_head(x, hp))


MODELS = {
    "lstm_vanilla": build_lstm_vanilla,
    "lstm_attention": build_lstm_attention,
    "gru_vanilla": build_gru_vanilla,
    "gru_attention": build_gru_attention,
    "tcn_vanilla": build_tcn_vanilla,
    "tcn_attention": build_tcn_attention,
}


# ----------------------------------------------------------------------- training
def _early_stop() -> keras.callbacks.EarlyStopping:
    return keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True
    )


def tune(name: str, ds: Dataset, max_trials: int = MAX_TRIALS) -> kt.HyperParameters:
    """Random search over the model's hyperparameters, selected on validation loss."""
    tuner = kt.RandomSearch(
        MODELS[name],
        objective="val_loss",
        max_trials=max_trials,
        executions_per_trial=1,
        seed=SEED,
        overwrite=True,
        directory=TUNING_DIR,
        project_name=name,
    )
    tuner.search(
        ds.x_train,
        ds.y_train,
        validation_data=(ds.x_val, ds.y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[_early_stop()],
        verbose=0,
    )
    return tuner.get_best_hyperparameters(num_trials=1)[0]


def train(name: str, hp: kt.HyperParameters, ds: Dataset):
    """Train a fresh model with the selected hyperparameters."""
    set_seed()
    model = MODELS[name](hp)
    history = model.fit(
        ds.x_train,
        ds.y_train,
        validation_data=(ds.x_val, ds.y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[_early_stop()],
        verbose=0,
    )
    MODELS_DIR.mkdir(exist_ok=True)
    model.save(MODELS_DIR / f"{name}.keras")
    return model, history


# --------------------------------------------------------------------- evaluation
def _inverse(ds: Dataset, arr: np.ndarray) -> np.ndarray:
    return ds.scaler.inverse_transform(arr.reshape(-1, 1)).ravel()


def predict_splits(model, ds: Dataset) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {split: (actual, predicted)} in original units (GWh)."""
    out = {}
    for split in ("train", "val", "test"):
        x, y = getattr(ds, f"x_{split}"), getattr(ds, f"y_{split}")
        out[split] = (_inverse(ds, y), _inverse(ds, model.predict(x, verbose=0)))
    return out


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAPE (%)": float(mean_absolute_percentage_error(actual, predicted) * 100),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def evaluate(name: str, hp: kt.HyperParameters, preds) -> pd.DataFrame:
    """Compute metrics for every split and save them to ``results/<name>.json``."""
    table = {split: metrics(*preds[split]) for split in preds}
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{name}.json", "w") as f:
        json.dump({"model": name, "hyperparameters": hp.values, "metrics": table}, f, indent=2)
    return pd.DataFrame(table).T.round(4)


def forecast(model, ds: Dataset, days: int = FORECAST_DAYS) -> pd.DataFrame:
    """Recursive multi-step forecast beyond the last observed date."""
    window = ds.last_window.copy()
    preds = []
    for _ in range(days):
        nxt = model.predict(window.reshape(1, LOOKBACK, 1), verbose=0)[0, 0]
        preds.append(nxt)
        window = np.append(window[1:], [[nxt]], axis=0)
    dates = pd.date_range(ds.dates.iloc[-1] + pd.Timedelta(days=1), periods=days, freq="D")
    return pd.DataFrame({"date": dates, "forecast_GWh": _inverse(ds, np.array(preds))})


# ------------------------------------------------------------------------ figures
def plot_series(data: pd.DataFrame) -> None:
    plt.figure(figsize=(15, 5))
    plt.plot(data["date"], data["GWh"])
    plt.title("Daily natural gas power generation, Mexico")
    plt.xlabel("Date")
    plt.ylabel("GWh")
    plt.tight_layout()
    plt.show()


def plot_learning_curve(history) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (scaled)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_predictions(name: str, ds: Dataset, preds, fc: pd.DataFrame | None = None) -> None:
    """Actual vs predicted over train/val/test (plus optional forecast); saved to results/figures."""
    split_dates = dict(zip(("train", "val", "test"), ds.split_dates))
    colors = {"train": "tab:green", "val": "tab:orange", "test": "tab:red"}
    labels = {"train": "Train", "val": "Validation", "test": "Test"}

    plt.figure(figsize=(16, 6))
    actual = np.concatenate([preds[s][0] for s in split_dates])
    plt.plot(pd.concat(split_dates.values()), actual, color="tab:blue", alpha=0.6, label="Actual")
    for split, dates in split_dates.items():
        plt.plot(dates, preds[split][1], color=colors[split], label=f"{labels[split]} prediction")
    for boundary in (split_dates["val"].iloc[0], split_dates["test"].iloc[0]):
        plt.axvline(boundary, color="gray", linestyle="--", linewidth=1)
    if fc is not None:
        plt.plot(fc["date"], fc["forecast_GWh"], color="black", linestyle="--",
                 label=f"{len(fc)}-day forecast")
    plt.title(f"{name.replace('_', ' ').upper()}: actual vs predicted")
    plt.xlabel("Date")
    plt.ylabel("GWh")
    plt.legend()
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / f"{name}.png", dpi=120)
    plt.show()
