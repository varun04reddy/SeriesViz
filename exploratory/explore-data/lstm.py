# ================================================================
#  LSTMTimeSeriesRegressor
#  ---------------------------------------------------------------
#  • Multivariate LSTM for forecasting 'spread' h steps ahead
#  • Handles:
#      – scaling (StandardScaler)
#      – sliding-window sampling
#      – chronological train/val/test split
#      – early stopping on validation MSE
#      – inverse-scaling of predictions
# ================================================================

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm  # nice progress bar in Jupyter


class _SeqDataset(Dataset):
    """Internal: turns full arrays into sliding-window sequences."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int, horizon: int):
        self.X, self.y = X, y
        self.seq_len, self.horizon = seq_len, horizon

    def __len__(self):
        return len(self.X) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]  # past window
        y_target = self.y[idx + self.seq_len + self.horizon - 1]  # future spread
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(
            y_target, dtype=torch.float32
        )


class _LSTMRegressor(nn.Module):
    """Internal: plain LSTM → FC → scalar."""

    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)   # out: [B, seq_len, hidden]
        return self.fc(out[:, -1]).squeeze(-1)  # take last timestep


class LSTMTimeSeriesRegressor:
    """
    Example
    -------
    >>> lstm = LSTMTimeSeriesRegressor(seq_len=60, horizon=10, epochs=30)
    >>> lstm.fit(dropped_temp)
    >>> lstm.evaluate()                # MSE on test set
    >>> preds = lstm.predict(dropped_temp)  # Series of predicted spreads
    """

    def __init__(
        self,
        seq_len: int = 60,
        horizon: int = 10,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.2,
        batch_size: int = 256,
        lr: float = 1e-3,
        epochs: int = 50,
        patience: int = 5,
        val_size: float = 0.2,
        test_size: float = 0.2,
        device: str | None = None,
    ):
        self.seq_len, self.horizon = seq_len, horizon
        self.hidden, self.layers, self.dropout = hidden, layers, dropout
        self.batch_size, self.lr = batch_size, lr
        self.epochs, self.patience = epochs, patience
        self.val_size, self.test_size = val_size, test_size
        self.scaler_X, self.scaler_y = StandardScaler(), StandardScaler()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: _LSTMRegressor | None = None
        self.feature_cols_: list[str] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame):
        """Chronological train/val/test split and training loop."""
        X, y, feat_cols = self._prepare_arrays(df)
        self.feature_cols_ = feat_cols

        # Split chronologically
        total = len(X)
        test_cut = int(total * (1 - self.test_size))
        val_cut = int(test_cut * (1 - self.val_size))

        X_train, y_train = X[:val_cut], y[:val_cut]
        X_val, y_val = X[val_cut:test_cut], y[val_cut:test_cut]
        X_test, y_test = X[test_cut:], y[test_cut:]
        self._test_true = y_test
        self.test_index_ = df.index[self.seq_len + self.horizon - 1 + test_cut :]

        # DataLoaders
        train_loader = DataLoader(
            _SeqDataset(X_train, y_train, self.seq_len, self.horizon),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            _SeqDataset(X_val, y_val, self.seq_len, self.horizon),
            batch_size=self.batch_size,
            shuffle=False,
        )
        self._test_loader = DataLoader(
            _SeqDataset(X_test, y_test, self.seq_len, self.horizon),
            batch_size=self.batch_size,
            shuffle=False,
        )

        # Model
        n_features = X.shape[1]
        self.model = _LSTMRegressor(
            n_features, self.hidden, self.layers, self.dropout
        ).to(self.device)
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        mse = nn.MSELoss()

        best_val, patience_left = np.inf, self.patience
        pbar = tqdm(range(1, self.epochs + 1), desc="Training")

        for epoch in pbar:
            # ----------------- training -----------------
            self.model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optim.zero_grad()
                loss = mse(self.model(xb), yb)
                loss.backward()
                optim.step()
                train_losses.append(loss.item())

            # ----------------- validation ----------------
            self.model.eval()
            with torch.no_grad():
                val_losses = [
                    mse(self.model(xb.to(self.device)), yb.to(self.device)).item()
                    for xb, yb in val_loader
                ]
            val_loss = np.mean(val_losses)
            pbar.set_postfix(Train_MSE=np.mean(train_losses), Val_MSE=val_loss)

            # ----------------- early stopping ------------
            if val_loss < best_val:
                best_val, best_state = val_loss, self.model.state_dict()
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left == 0:
                    print("Early stopping triggered")
                    break

        # Load best weights
        self.model.load_state_dict(best_state)

    def evaluate(self) -> float:
        """Return MSE on the held-out test set."""
        preds = self._predict_loader(self._test_loader)
        true = self._test_true[self.seq_len + self.horizon - 1 :]
        mse = mean_squared_error(true, preds)
        print(f"Test MSE: {mse:.4f}")
        return mse

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict future spread for **df** (needs enough history)."""
        if self.model is None:
            raise RuntimeError("Call .fit() first.")

        X, _, _ = self._prepare_arrays(df, fit=False)
        loader = DataLoader(
            _SeqDataset(X, np.zeros_like(X[:, 0]), self.seq_len, self.horizon),
            batch_size=self.batch_size,
            shuffle=False,
        )
        preds = self._predict_loader(loader)
        index = df.index[self.seq_len + self.horizon - 1 :]
        return pd.Series(preds, index=index, name="predicted_spread")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_arrays(
        self, df: pd.DataFrame, *, fit: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, list[str]]:
        """Return scaled feature & target arrays."""
        feature_cols = (
            ["nifty_returns", "es_returns", "expected_nifty_returns", "spread"]
            + [c for c in df.columns if c.startswith("rolling_beta")]
        )
        df = df[feature_cols].dropna()

        X = df.drop(columns=["spread"]).values
        y = df["spread"].values.reshape(-1, 1)

        if fit:
            X = self.scaler_X.fit_transform(X)
            y = self.scaler_y.fit_transform(y)
        else:
            X = self.scaler_X.transform(X)
            y = self.scaler_y.transform(y)

        return X.astype(np.float32), y.squeeze(-1).astype(np.float32), feature_cols

    def _predict_loader(self, loader: DataLoader) -> np.ndarray:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                out = self.model(xb.to(self.device)).cpu().numpy()
                preds.append(out)
        preds = np.concatenate(preds)
        return self.scaler_y.inverse_transform(preds.reshape(-1, 1)).squeeze()
