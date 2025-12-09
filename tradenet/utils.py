import os
from typing import Dict
import numpy as np
import pandas as pd
from .config import RAW_DATA_DIR

# load the raw data for a given ticker as a DataFrame
def load_raw_ticker_data(ticker):
    file_path = os.path.join(RAW_DATA_DIR, f"{ticker.upper()}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file for ticker {ticker} not found at {file_path}")

    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df.sort_values("Date").reset_index(drop=True)
    
    if "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"].astype(float)
    else:
        df["Close"] = df["Close"].astype(float)
    
    df = df.dropna(subset=["Close"]).reset_index(drop=True)

    return df


# Feature engineering, add technical indicators
def preprocess_raw_data(df):

    if "Close" not in df.columns:
        raise KeyError("Input DataFrame must contain 'Close' column.")
    
    df = df.copy()

    # Log returns: r_t = log(P_t / P_{t-1})
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["log_return"] = df["log_return"].fillna(0.0)

    # Moving averages
    df["ma20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    df["ma50"] = df["Close"].rolling(window=50, min_periods=1).mean()

    # Volatility (rolling std dev of log returns)
    df["vol20"] = df["log_return"].rolling(window=20, min_periods=1).std().fillna(0.0)

    # Backfill any missing values in moving averages
    df[["ma20", "ma50"]] = df[["ma20", "ma50"]].bfill()

    return df


# Add a window of past log returns as features
def add_return_trend(df, window =  20):
    df = df.copy()
    
    if "log_return" not in df.columns:
        raise KeyError("Input DataFrame must contain 'log_return' column.")

    for i in range(window):
        df[f"trend_{i}"] = df["log_return"].shift(i)

    # Replace any initial NaNs from shifting with zeros
    df = df.fillna(0.0)

    return df


# Split data into train, validation, and test sets based on date ranges
def train_test_val_split(df):
    
    if "Date" not in df.columns:
        raise KeyError("Input DataFrame must contain 'Date' column for  train/test split.")

    df = df.sort_values("Date").reset_index(drop=True)

    df_train = df[(df["Date"] >= "2015-01-01") & (df["Date"] <= "2019-12-31")].reset_index(drop=True)
    df_val = df[(df["Date"] >= "2020-01-01") & (df["Date"] <= "2022-12-31")].reset_index(drop=True)
    df_test = df[(df["Date"] >= "2023-01-01") & (df["Date"] <= "2024-12-31")].reset_index(drop=True)

    if len(df_train) == 0 or len(df_val) == 0 or len(df_test) == 0:
        raise ValueError(
            "One of the splits is empty. Check that the input DataFrame covers "
            "2014-01-01 to 2022-12-31."
        )

    return {"train_df": df_train, "val_df": df_val, "test_df": df_test}
