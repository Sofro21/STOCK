from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

df = pd.read_csv("proc_appl.csv", index_col=False)
mses = []
trues = []
preds = []
i = 0
counter = 0
while counter < 60:

    df["Date"] = pd.to_datetime(df["Date"])

    train_start_date = pd.to_datetime("2016-01-01") + pd.Timedelta(days=i)
    train_end_date = pd.to_datetime("2017-01-03") + pd.Timedelta(days=i)

    test_date = pd.to_datetime("2017-01-03") + pd.Timedelta(days=i)

    train = df[(df["Date"] > train_start_date) & (df["Date"] < train_end_date)][
        ["Date", "Close"]
    ]
    test = df[df["Date"] == test_date][["Date", "Close"]]

    X_train = np.array(range(len(train)))  # Using indices as features
    y_train = train["Close"].values.reshape(-1, 1)

    X_test = np.array(
        range(len(train), len(train) + len(test))
    )  # Using indices for simplicity
    y_test = test["Close"].values.reshape(-1, 1)

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

    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train the model
    model.fit(X_train_scaled, y_train_scaled, epochs=27, batch_size=30)

    # Make predictions on the test set
    y_pred_scaled = model.predict(X_test_scaled)

    # Inverse transform the predictions
    y_pred = scaler.inverse_transform(y_pred_scaled[0])

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    mses.append(mse)
    counter += 1
    i += 1
    trues.append(y_test[0])
    preds.append(y_pred[0])

plt.plot(trues, c="green")
plt.plot(preds, c="blue")
plt.show()

# Plot the results
# plt.scatter(X_train, y_train, c="green", label="Train")
# plt.scatter(X_test, y_test, c="blue", label="Actual Test")
# plt.scatter(X_test, y_pred, c="red", label="Predicted Test")
# plt.legend()
# plt.show()
#
