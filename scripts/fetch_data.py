import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from tradenet.data_loader import YahooDownloader

def main():
    parser = argparse.ArgumentParser(
        description="Fetch historical OHLCV data from Yahoo Finance."
    )

    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Ticker symbol (e.g., AAPL)",
    )

    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date in YYYY-MM-DD format",
    )

    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date in YYYY-MM-DD format",
    )

    args = parser.parse_args()

    loader = YahooDownloader(
        
        ticker=args.ticker.upper(),
        start=args.start,
        end=args.end,
    )

    loader.fetch_and_save()


if __name__ == "__main__":
    main()
