"""
Downloads Reliance Industries data from Yahoo Finance using yfinance,
cleans it, and saves it as datanewfinal.csv

Usage:
    python preprocess.py                          # download from Yahoo Finance (default)
    python preprocess.py --input rw.csv           # use existing CSV file instead
    python preprocess.py --output data/out.csv    # custom output path
"""
import argparse
import os
import pandas as pd

def download_from_yahoo(output_path):
    try:
        import yfinance as yf
    except ImportError:
        print("[ERROR] yfinance not installed. Run: pip install yfinance")
        return False

    print("[INFO] Downloading RELIANCE.NS from Yahoo Finance...")
    ticker = yf.Ticker("RELIANCE.NS")
    df = ticker.history(start="2021-03-01", end="2025-02-26")

    if df.empty:
        print("[ERROR] No data returned from Yahoo Finance.")
        return False

    df = df.reset_index()
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    print(f"[INFO] Downloaded {len(df)} rows  ({df['Date'].min()} → {df['Date'].max()})")
    return df

def clean(df, output_path):
    numeric_cols = ["Close", "High", "Low", "Open", "Volume"]

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    before = len(df)
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[~((df[numeric_cols] < lower) | (df[numeric_cols] > upper)).any(axis=1)]
    after = len(df)

    print(f"[INFO] Outliers removed: {before - after} rows dropped")
    print(f"[INFO] Missing values remaining: {df.isnull().sum().sum()}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved cleaned data → {output_path}  ({after} rows)")
    return True

def preprocess_from_csv(input_path, output_path):
    print(f"[INFO] Reading from {input_path}...")
    df = pd.read_csv(input_path)

    # handle raw Yahoo Finance CSV which has 2 metadata rows at top
    if df.columns[0] == "Price":
        df = df.loc[2:].reset_index(drop=True)
        df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

    df["Date"] = pd.to_datetime(df["Date"])
    return clean(df, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=None,                    help="Path to raw CSV (skip to download from Yahoo)")
    parser.add_argument("--output", default="data/datanewfinal.csv", help="Output path for cleaned CSV")
    args = parser.parse_args()

    if args.input:
        preprocess_from_csv(args.input, args.output)
    else:
        df = download_from_yahoo(args.output)
        if df is not False:
            clean(df, args.output)