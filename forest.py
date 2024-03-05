# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv("proc_appl.csv", index_col=False)

df["Date"] = pd.to_datetime(df["Date"])

df["Month"] = df["Date"].dt.month

train = df[(df["Date"] > "2010-01-01") & (df["Date"] < "2016-01-01")][["Month", "FEMA"]]
test = df[df["Date"] > "2016-01-01"][["Month", "FEMA"]]

X_train, y_train = train.drop("FEMA", axis=1), train["FEMA"]
X_test, y_test = test.drop("FEMA", axis=1), test["FEMA"]

# %%

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

for i in range(len(y_pred)):
    y_pred[i] *= 2 ** ((10 + i) / 500)

mse = mean_squared_error(y_test, y_pred)
print(mse)

print(y_pred)

plt.plot(range(len(y_test)), y_test, c="RED")
plt.plot(range(len(y_test)), y_pred, c="Blue")
plt.show()


# Load the dataset
# df = pd.read_csv("proc_csvs/a.csv")
#
# # Convert the "Date" column to a pandas datetime object
# df["Date"] = pd.to_datetime(df["Date"])
#
# # Extract the month from the "Date" column and create a new "Month" feature
# df["Month"] = df["Date"].dt.month
#
# # Select features (X) and target variable (y)
# features = df.drop(
#     ["Date", "Close", "SMA_200"], axis=1
# )  # Exclude non-numeric and target columns
# target = df["Close"]
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     features, target, test_size=0.2, random_state=42
# )
#
#
# rf_model = RandomForestRegressor(random_state=42)
#
# # Train the model
# rf_model.fit(X_train, y_train)
#
# # Specify the date for which you want to make a prediction
# input_date = "2024-03-03"  # Replace with the desired date
# input_features = df[df["Date"] == input_date].drop(["Date", "Close", "SMA_200"], axis=1)
#
# # Make a prediction for the specified date
# predicted_price = rf_model.predict(input_features)
#
# print(f"Predicted Close Price on {input_date}: {predicted_price[0]}")

# %%
