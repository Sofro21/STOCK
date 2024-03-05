import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("proc_csvs/aal.csv", index_col=False)

# Convert the "Date" column to a pandas datetime object
df["Date"] = pd.to_datetime(df["Date"])

# Extract the month from the "Date" column and create a new "Month" feature
df["Month"] = df["Date"].dt.month

# Select features (X) and target variable (y)
features = df.drop(
    ["Date", "Close", "SMA_200"], axis=1
)  # Exclude non-numeric and target columns
target = df["Close"]

# Split the data into training and testing sets
"""X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.8, random_state=42
)"""

train = df[df["Date"] < "2016-01-01"].drop(["Date", "SMA_200"], axis=1)

test = df[df["Date"] >= "2016-01-01"].drop(["Date", "SMA_200"], axis=1)

y_train = train["Close"]
X_train = train.drop(["Close", "Open", "High"], axis=1)


X_test = test.drop(["Close", "Open", "High"], axis=1)
y_test = test["Close"]


# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the Neural Network model
model = Sequential()
model.add(Dense(64, activation="relu", input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="linear"))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")

# Train the model
model.fit(
    X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=2
)

# Evaluate the model on the test set
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse}")

plt.scatter(range(len(y_test)), y_test, c="Blue")
plt.scatter(range(len(y_pred)), y_pred, c="Red")
plt.show()


exit()
# Example for making a prediction using the trained model
# Specify input parameters for prediction (same features as used during training)
input_parameters = {
    "Open": 18.0,
    "High": 18.5,
    "Low": 17.8,
    "Volume": 4000000,
    # Add values for other features...
}

# Transform and scale the input parameters
input_data = scaler.transform(pd.DataFrame([input_parameters]))

# Make a prediction using the trained model
predicted_price = model.predict(input_data)

print(f"Predicted Close Price: {predicted_price[0, 0]}")
