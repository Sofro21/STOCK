# %%
import pandas as pd
from glob import glob
import os
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volume import VolumeWeightedAveragePrice
import os

# %%

# %%

# %%


def preprocess_data(file_path):

    df = pd.read_csv(file_path, index_col=False)

    df.fillna(0, inplace=True)

    return df.drop(["Stock Splits", "Dividends"], axis=1)


# %%
def concatenate_dataframes(file_list, dir):
    dfs = []
    for file in file_list:
        df = pd.read_csv(os.path.join(dir, file))
        df["Company"] = file.split(".")[0]

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# %%


def calculate_macd(df):
    close_prices = df["Close"]
    macd = MACD(close=close_prices, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    return df


# %%


def calculate_bollinger(df):
    close_prices = df["Close"]

    bollinger = BollingerBands(close=close_prices, window=20, window_dev=2)

    df["bb_bbm"] = bollinger.bollinger_mavg()
    df["bb_bbh"] = bollinger.bollinger_hband()
    df["bb_bbl"] = bollinger.bollinger_lband()

    return df


# %%
def calculate_rsi(df):
    close_prices = df["Close"]

    rsi = RSIIndicator(close=close_prices, window=14)

    df["rsi"] = rsi.rsi()

    return df


# %%


def calculate_vwap(df):
    high_prices = df["High"]
    low_prices = df["Low"]
    close_prices = df["Close"]
    volume = df["Volume"]

    # Calculate VWAP
    vwap = VolumeWeightedAveragePrice(
        high=high_prices, low=low_prices, close=close_prices, volume=volume
    )

    # Add VWAP column to your DataFrame
    df["vwap"] = vwap.volume_weighted_average_price()

    return df


# %%
def add_lags(df, num_lags):
    for i in range(1, num_lags + 1):
        try:
            df[f"Close_Lag_{i}"] = df["Close"].shift(i)
        except:
            df[f"Close_Lag_{i}"] = 0

    return df


# %%
def add_MAs(df):
    sma_windows = [5, 10, 20, 50, 100, 200]
    ema_windows = [5, 10, 20, 50, 100, 200]
    for window in sma_windows:
        df[f"SMA_{window}"] = df["Close"].rolling(window=window).mean()

    # Calculate EMA
    for window in ema_windows:
        df[f"EMA_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()

    return df


# %%
directory = "csvs"
target = "proc_csvs"

for file in os.listdir(directory):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(directory, file))
        df = df[df["Date"] >= "2009-01-01"]
        df = df.drop("Dividends", axis=1)
        df = df.drop("Stock Splits", axis=1)
        df = calculate_macd(df)
        df = calculate_bollinger(df)
        df = calculate_rsi(df)
        df = calculate_vwap(df)
        df = add_lags(df, 5)
        df = add_MAs(df)
        df = df.iloc[192:, :]
        df.to_csv(target + "/" + file, index=False)


# %%
def calculate_EMAs(df):
    const = 0.15
    closing_prices = df["Close"]
    avg = sum(closing_prices[:6]) / 7
    df["FEMA"] = [
        ((curr_closing_price - avg) * const) + avg
        for curr_closing_price in closing_prices
    ]

    const = 0.07
    avg = sum(closing_prices[:25]) / 26
    df["SEMA"] = [
        ((curr_closing_price - avg) * const) + avg
        for curr_closing_price in closing_prices
    ]

    df["DIFF"] = df["FEMA"] - df["SEMA"]

    avg = sum(df["DIFF"][26:35]) / 9
    const = 0.2
    df["SIGNAL"] = ((df["DIFF"] - avg) * const) + avg

    df["HISTOGRAM"] = df["DIFF"] - df["SIGNAL"]

    return df[36:]


df = pd.read_csv("snp500_data/General Motors_GM_data.csv", index_col=False)

df = calculate_EMAs(df)

df.to_csv("snp500_data/General Motors_GM_data.csv", index=False)


# %%

# %%
