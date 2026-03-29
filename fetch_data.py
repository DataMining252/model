import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime

START = "2004-01-01"
END = datetime.today().strftime("%Y-%m-%d")


def fetch_yfinance(symbol, name):
    df = yf.download(symbol, start=START, end=END)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].rename(columns={"Close": name})
    df.index.name = "Date"
    return df


def fetch_fred(series, name):
    df = pdr.DataReader(series, "fred", START, END)
    df = df.rename(columns={series: name})
    df.index.name = "Date"
    return df


def main():
    print("Fetching data...")

    # Market data
    dxy = fetch_yfinance("DX-Y.NYB", "DXY")
    sp500 = fetch_yfinance("^GSPC", "SP500")
    oil = fetch_yfinance("CL=F", "OIL")

    # Macro data
    rate = fetch_fred("FEDFUNDS", "INTEREST_RATE")
    cpi = fetch_fred("CPIAUCSL", "CPI")

    # Merge all
    df = dxy.join(sp500, how="outer")
    df = df.join(oil, how="outer")
    df = df.join(rate, how="outer")
    df = df.join(cpi, how="outer")

    df = df.sort_index()

    # tạo full daily index
    df = df.asfreq("D")

    # forward fill tất cả
    df = df.ffill()

    # drop NaN đầu
    df = df.dropna()

    # Save
    df.to_csv("raw/macro_data.csv")

    print("Saved to macro_data.csv")
    print(df.head())


if __name__ == "__main__":
    main()