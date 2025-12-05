import argparse
import os
import yfinance as yf
import pandas as pd

from tradenet.config import RAW_DATA_DIR

# Fetch historical OHLCV data from Yahoo Finance and save as CSV
def fetch_and_save(ticker: str, start: str, end: str):

    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}. Check the symbol or date range.")

    df.reset_index(inplace=True)
    df.rename(columns=str.capitalize, inplace=True)

    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    save_path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")

    df.to_csv(save_path, index=False)

    print(f"Downloaded {ticker} data: {len(df)} rows saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical OHLCV data from Yahoo Finance.")

    parser.add_argument("--ticker", type=str, required=True,
                        help="Ticker symbol (e.g., AAPL)")
    parser.add_argument("--start", type=str, required=True,
                        help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end", type=str, required=True,
                        help="End date in YYYY-MM-DD format")

    args = parser.parse_args()

    fetch_and_save(
        ticker=args.ticker.upper(),
        start=args.start,
        end=args.end
    )
