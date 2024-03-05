# importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# creating dataframe

df = pd.read_csv("aapl.csv")
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0, len(df)), columns=["Date", "Close"])
for i in range(0, len(data)):
    new_data["Date"][i] = data["Date"][i]
    new_data["Close"][i] = data["Close"][i]

# setting index
new_data = new_data[new_data["Date"] > "2010-01-01"]
new_data.index = new_data.Date
new_data.drop("Date", axis=1, inplace=True)

# creating train and test sets
dataset = new_data.values
print("\n\n\n\n This is the fucnking nswer you moron: ", type(dataset))
train = dataset[0:987, :]
# arr = np.array(['2020-06-06', '2020-06-07'], dtype='datetime64')
# arr2 = arr.astype('datetime64[D]')
# arr3 = np.datetime64(arr, 'D')
# Date to add
# arr4 = np.datetime64('2020-06-06')

# Append the new date to the existing array

valid = dataset[987:, :]
# valid = np.append(valid, arr4)
# converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60, len(train)):
    x_train.append(scaled_data[i - 60 : i, 0])
    y_train.append(scaled_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

# new_dates = pd.DataFrame()

# temp = []

# for j in range(3):
#     for i in range(1, 31):
#         temp.append(f'2020-0{j+6}-{"0" + str(i) if i < 10 else i}')


# new_dates["Date"] = temp
# new_dates["Date"] = pd.to_datetime(new_dates["Date"])

# Assuming arr is your NumPy array with datetime64 dtype
# arr = np.array(new_dates["Date"], dtype='datetime64')
# print(arr)

# Convert datetime64 to int (Unix timestamp in seconds)
# arr_as_int = (arr - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
# print(arr_as_int)

# new_dates["Days"] = (new_dates["Date"] - new_dates.index.min()).dt.days

# # Scale the "Days" column
# new_dates["Days"] = scaler.transform(np.array(new_dates["Days"]).reshape(-1, 1))

# # Reshape input data for the LSTM model
# new_dates = np.reshape(new_dates["Days"].values, (new_dates["Days"].shape[0], 1, 1))

model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

# predicting 246 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60 :].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i - 60 : i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)


train = new_data[:987]
valid = new_data[987:]
valid["Predictions"] = closing_price
plt.plot(train["Close"], c="Blue")
plt.plot(valid["Predictions"], c="Red")
plt.plot(valid["Close"], c="Pink")
plt.show()
