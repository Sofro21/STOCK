from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

df = pd.read_csv("snp500_data/General Motors_GM_data.csv", index_col=False)
i = 0
counter = 0
while counter < 10:
    df["Date"] = pd.to_datetime(df["Date"])
    train_start_date = pd.to_datetime("2016-01-01") + pd.Timedelta(days=i, unit="D")
    train_end_date = pd.to_datetime("2024-03-06") + pd.Timedelta(days=i, unit="D")

    test_date = pd.to_datetime("2024-03-06") + pd.Timedelta(days=i, unit="D")

    temp = pd.DataFrame()
    temp["Date"] = [test_date]
    temp["FEMA"] = [0.0]
    temp["SEMA"] = [0.0]

    df = df._append(temp, ignore_index=True)

    train = df[(df["Date"] > train_start_date) & (df["Date"] < train_end_date)][
        ["Date", "FEMA"]
    ]
    test = df[df["Date"] == test_date][["Date", "FEMA"]]

    X_train = np.array(range(len(train)))  # Using indices as features
    y_train = train["FEMA"].values.reshape(-1, 1)

    X_test = np.array(
        range(len(train), len(train) + len(test))
    )  # Using indices for simplicity
    y_test = test["FEMA"].values.reshape(-1, 1)

    if not len(X_test) or not len(X_train):
        i += 1
        continue

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1))
    X_test_scaled = scaler.transform(X_test.reshape(-1, 1))
    y_train_scaled = scaler.fit_transform(y_train)
    y_test_scaled = scaler.transform(y_test)

    # Reshape data for LSTM
    X_train_scaled = X_train_scaled.reshape(
        (X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    )
    X_test_scaled = X_test_scaled.reshape(
        (X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
    )

    # Build LSTM model
    model = Sequential()
    model.add(
        LSTM(
            units=50,
            return_sequences=True,
            input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
        )
    )
    model.add(Dropout(rate=0.25))
    model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=0.00013141592), loss="mean_squared_error"
    )

    # Train the model
    model.fit(X_train_scaled, y_train_scaled, epochs=27, batch_size=30)

    # Make predictions on the test set
    y_pred_scaled = model.predict(X_test_scaled)

    # Inverse transform the predictions
    y_pred = scaler.inverse_transform(y_pred_scaled[0])
    df.at[df.index[-1], "FEMA"] = y_pred[0][0]

    train = df[(df["Date"] > train_start_date) & (df["Date"] < train_end_date)][
        ["Date", "SEMA"]
    ]
    test = df[df["Date"] == test_date][["Date", "SEMA"]]

    X_train = np.array(range(len(train)))  # Using indices as features
    y_train = train["SEMA"].values.reshape(-1, 1)

    X_test = np.array(
        range(len(train), len(train) + len(test))
    )  # Using indices for simplicity
    y_test = test["SEMA"].values.reshape(-1, 1)

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1))
    X_test_scaled = scaler.transform(X_test.reshape(-1, 1))
    y_train_scaled = scaler.fit_transform(y_train)
    y_test_scaled = scaler.transform(y_test)

    # Reshape data for LSTM
    X_train_scaled = X_train_scaled.reshape(
        (X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    )
    X_test_scaled = X_test_scaled.reshape(
        (X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
    )

    model.fit(X_train_scaled, y_train_scaled, epochs=27, batch_size=30)
    y_pred_scaled = model.predict(X_test_scaled)

    y_pred = scaler.inverse_transform(y_pred_scaled[0])

    df.at[df.index[-1], "SEMA"] = y_pred[0][0]

    counter += 1
    i += 1
    df.at[df.index[-1], "SEMA"] = y_pred[0][0]
    df.to_csv("snp500_data/General Motors_GM_data.csv", index=False)


# Plot the results
# plt.scatter(X_train, y_train, c="green", label="Train")
# plt.scatter(X_test, y_test, c="blue", label="Actual Test")
# plt.scatter(X_test, y_pred, c="red", label="Predicted Test")
# plt.legend()
# plt.show()
#
