import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load the time series data
data = pd.read_csv("NVDA_stock_prices.csv")

# convert date column to numerical format
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].astype(np.int64) // 10**9

# Scale the data between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
feature_cols = data.columns.difference(['Date'])
data[feature_cols] = scaler.fit_transform(data[feature_cols])

# Split the data into training and test sets
train_size = int(len(data) * 0.8)
test_size = len(data) - train_size
data = data.values
train, test = data[0:train_size,:], data[train_size:len(data),:]

# Convert data into numpy arrays
X_train, y_train = train[:, 1:], train[:, 0]
X_test, y_test = test[:, 1:], test[:, 0]

# reshape data
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# Create the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred = np.squeeze(y_pred)

# Print out the results
print("Predicted stock prices:", y_pred)
print("Actual stock prices:", y_test)

# Plot the results
plt.plot(y_pred, label='Predicted')
plt.plot(y_test, label='Actual')
plt.legend()
plt.show()