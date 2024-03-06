# %%
import pandas as pd
from glob import glob
import os


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


def main(file_path):
    df = pd.read_csv(file_path, index_col=False)

    df = calculate_EMAs(df)

    df.to_csv(file_path, index=False)


# %%

# %%
