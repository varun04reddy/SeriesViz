# A reusable class to train and test an XGBoost classifier that predicts mean‑reversion
# between Nifty and ES Mini based on rolling betas, returns, and spread.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt


class TrainTestXGBoost:
    """
    Train and evaluate an XGBoost classifier for Nifty/ES reversion prediction.

    Parameters
    ----------
    label_horizon : int, default 3
        Number of future periods over which to test for mean‑reversion.
    test_size : float, default 0.2
        Fraction of the dataset to allocate to the test split.
    shuffle : bool, default False
        Whether to shuffle when splitting. For time‑series keep this False.
    xgb_params : dict, optional
        Additional parameters to pass to XGBClassifier.

    Usage
    -----
    >>> model = TrainTestXGBoost(label_horizon=3)
    >>> model.fit(df)            # df contains timestamp, returns, betas, spread
    >>> model.evaluate()         # prints classification metrics
    >>> preds = model.predict(df)  # get probabilities for new data
    >>> model.plot_feature_importance()
    """

    def __init__(self,
                 label_horizon: int = 3,
                 test_size: float = 0.2,
                 shuffle: bool = False,
                 xgb_params: dict | None = None):
        self.label_horizon = label_horizon
        self.test_size = test_size
        self.shuffle = shuffle
        default_params = {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "objective": "binary:logistic",
            "eval_metric": "logloss"
        }
        if xgb_params:
            default_params.update(xgb_params)
        self.model = XGBClassifier(**default_params)
        self.feature_cols_: list[str] | None = None
        self.X_train_: pd.DataFrame | None = None
        self.X_test_: pd.DataFrame | None = None
        self.y_train_: pd.Series | None = None
        self.y_test_: pd.Series | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame) -> "TrainTestXGBoost":
        """
        Prepares data, trains the XGBoost model, and stores train/test splits.
        """
        X, y = self._prepare_features_and_label(df.copy())

        # Train‑test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, shuffle=self.shuffle)
        
        self.test_index_ = X_test.index

        # Save splits for later use
        self.X_train_, self.X_test_ = X_train, X_test
        self.y_train_, self.y_test_ = y_train, y_test

        # Fit model
        self.model.fit(X_train, y_train)
        return self

    def predict(self, df: pd.DataFrame, proba: bool = True):
        """
        Generate predictions or probabilities for new data.
        """
        if self.feature_cols_ is None:
            raise ValueError("Model must be fitted before calling predict.")

        X = df[self.feature_cols_].dropna()
        return self.model.predict_proba(X)[:, 1] if proba else self.model.predict(X)

    def evaluate(self) -> None:
        """
        Prints classification metrics on the held‑out test set.
        """
        if self.X_test_ is None or self.y_test_ is None:
            raise ValueError("Model must be fitted before evaluation.")

        y_pred = self.model.predict(self.X_test_)
        print("Confusion Matrix:\n", confusion_matrix(self.y_test_, y_pred))
        print("\nClassification Report:\n", classification_report(self.y_test_, y_pred))

    def plot_feature_importance(self, max_features: int = 15) -> None:
        """
        Displays feature importance based on gain.
        """
        plot_importance(self.model, max_num_features=max_features, importance_type="gain")
        plt.title("XGBoost Feature Importance (gain)")
        plt.show()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _prepare_features_and_label(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Creates engineered features and target label.
        """
        # Ensure essential columns exist
        required_cols = {"nifty_returns", "es_returns", "spread"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # ---------- Label creation ---------- #
        horizon = self.label_horizon
        future_ret = df["nifty_returns"].shift(-horizon).rolling(horizon).sum()
        df["label"] = ((future_ret * df["spread"]) < 0).astype(int)

        # ---------- Feature selection ---------- #
        beta_cols = [col for col in df.columns if col.startswith("rolling_beta")]
        self.feature_cols_ = beta_cols + [
            "nifty_returns", "es_returns", "spread", "expected_nifty_returns"
        ]

        # Drop rows with NaNs in features or label
        df = df.dropna(subset=self.feature_cols_ + ["label"])

        X = df[self.feature_cols_]
        y = df["label"]

        return X, y
