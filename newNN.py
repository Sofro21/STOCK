import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def main2(file_path):
    def proc(df):

        dt = df[df["DIFF"].isnull()]
        print(dt)

        dt["DIFF"] = dt["FEMA"] - dt["SEMA"]

        avg = sum(dt["DIFF"][26:34]) / 9
        const = 0.2
        dt["SIGNAL"] = ((dt["DIFF"] - avg) * const) + avg

        dt["HISTOGRAM"] = dt["DIFF"] - dt["SIGNAL"]

        start_date = dt["Date"].iloc[0]
        test = df[df["Date"] >= start_date]
        test["DIFF"] = dt["DIFF"]
        test["SIGNAL"] = dt["SIGNAL"]
        test["HISTOGRAM"] = dt["HISTOGRAM"]

        df.iloc[-test.shape[0] :] = test.values

        return df

    df = pd.read_csv(file_path, index_col=False)
    df = proc(df)
    df["Date"] = pd.to_datetime(df["Date"])

    train = df[(df["Date"] >= "2016-01-01") & (df["Date"] < "2024-03-06")].drop(
        ["Date"], axis=1
    )
    test = df[df["Date"] >= "2024-03-06"].drop(["Date"], axis=1)

    X_train = train.drop("Close", axis=1)
    y_train = train["Close"]

    X_test = test.drop("Close", axis=1)
    y_test = test["Close"]

    model = Sequential()

    model.add(InputLayer(input_shape=(X_train.shape[1],)))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(rate=0.23141592))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="linear"))

    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")

    model.fit(X_train, y_train, epochs=10, batch_size=32)

    y_pred = model.predict(X_test)

    df.loc[df["Date"] >= "2024-03-06", "Close"] = (
        y_pred.flatten() * 23.1406926328 * 1.61803398875
    )
    df.to_csv(file_path, index=False)
