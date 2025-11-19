import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader
from .dataset import TimeSeriesDataset


class DataModule:
    def __init__(self, cfg):
        self.cfg = cfg
        self.scaler = None

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.cfg.csv_path, parse_dates=True)
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna()
        return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        df["log_volume"] = np.log(df["Volume"] + 1)

        df["high_low_ratio"] = df["High"] / df["Low"]
        df["close_open_ratio"] = df["Close"] / df["Open"]
        df["price_range"] = (df["High"] - df["Low"]) / df["Close"]
        df["body_ratio"] = abs(df["Close"] - df["Open"]) / (
            df["High"] - df["Low"] + 1e-10
        )

        for window in [5, 10, 20, 50]:
            df[f"sma_{window}"] = df["Close"].rolling(window=window).mean()
            df[f"price_to_sma_{window}"] = df["Close"] / df[f"sma_{window}"]

        return df

    def create_sequence(
        self, data: np.ndarray, target_col_idx: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []

        for i in range(len(data) - self.cfg.window_size):
            X.append(data[i : i + self.cfg.window_size])
            y.append(data[i + self.cfg.window_size, target_col_idx])

        return np.array(X), np.array(y)

    def fit_scaler(self, data: np.ndarray) -> None:
        if self.cfg.scaler_type == "minmax":
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif self.cfg.scaler_type == "standard":
            self.scaler = StandardScaler()

        self.scaler.fit(data)

    def transform_data(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data)

    def get_scaler(self):
        return self.scaler

    def split_data(
        self, df: pd.DataFrame, val_size: int, test_size: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        total_samples = len(df)

        test_samples = int(total_samples * test_size)
        val_samples = int(total_samples * val_size)
        train_samples = total_samples - test_samples - val_samples

        train_end = train_samples
        val_start = train_end
        val_end = val_start + val_samples
        test_start = val_end

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[val_start:val_end].copy()
        test_df = df.iloc[test_start:].copy()

        return train_df, val_df, test_df

    def get_loaders(
        self, val_size: int = 0.1, test_size: int = 0.1
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        df = self.load_data()
        df = self.prepare_features(df)
        df = self.clean_data(df)

        train_df, val_df, test_df = self.split_data(df, val_size, test_size)

        feature_cols = [col for col in df if col not in (self.cfg.exclude_cols or [])]
        scale_cols = [
            col for col in feature_cols if col not in (self.cfg.no_scale_cols or [])
        ]

        self.fit_scaler(train_df[scale_cols].values)

        # df[scale_cols] = self.transform_data(df[scale_cols].values)
        train_arr = self.transform_data(train_df[scale_cols].values)
        val_arr = self.transform_data(val_df[scale_cols].values)
        test_arr = self.transform_data(test_df[scale_cols].values)

        target_idx = feature_cols.index(self.cfg.target_col)

        X_train, y_train = self.create_sequence(train_arr, target_col_idx=target_idx)
        X_val, y_val = self.create_sequence(val_arr, target_col_idx=target_idx)
        X_test, y_test = self.create_sequence(test_arr, target_col_idx=target_idx)

        train_ds = TimeSeriesDataset(X_train, y_train)
        val_ds = TimeSeriesDataset(X_val, y_val)
        test_ds = TimeSeriesDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=self.cfg.batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds, batch_size=self.cfg.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, self.scaler
