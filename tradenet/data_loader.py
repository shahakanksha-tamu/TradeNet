import os
import pandas as pd
import yfinance as yf
from typing import Optional
from tradenet.config import RAW_DATA_DIR

class YahooDownloader:

    def __init__(self, ticker: str, start: str, end: str):
        self.ticker = ticker.upper()
        self.start = start
        self.end = end

    def fetch(self):
        df = yf.download(self.ticker, start=self.start, end=self.end, progress=False)

        if df.empty:
            raise ValueError(
                f"No data returned for ticker {self.ticker}. "
                f"Check the ticker symbol or date range."
            )

        df.reset_index(inplace=True)
        df.rename(columns=str.capitalize, inplace=True)

        return df

    def fetch_and_save(self, save_dir: Optional[str] = None) -> str:
        
        df = self.fetch()
        if save_dir is None:
            save_dir = RAW_DATA_DIR

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{self.ticker}.csv")
        df.to_csv(save_path, index=False)

        print(f"{self.ticker}: Downloaded {len(df)} rows and saved to {save_path}")
        return save_path
