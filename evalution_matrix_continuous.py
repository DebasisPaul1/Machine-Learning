import pandas as pd
import numpy as np

mydata = pd.read_csv("House Price Prediction Dataset.csv")

X = np.array(mydata[["Area", "Bedrooms", "Bathrooms", "Floors"]])
y = np.array(mydata["Price"])

print(X)
print(y)

m, n = X.shape

W = np.array([0, 0, 0, 0], dtype=float)
b = 0

alpha = 0.0000001
epochs = 100000

for i in range(epochs):

    y_pred = np.dot(X, W) + b
    error = y_pred - y

    dW = (2/m) * np.dot(X.T, error)
    db = (2/m) * np.sum(error)

    W = W - alpha * dW
    b = b - alpha * db

print("W =", W)
print("b =", b)

X_Test = np.array([1370, 3, 4, 1])

Y_pred = np.dot(X_Test, W) + b

print("Predicted Price =", Y_pred)

MAE = np.mean(np.abs(y - y_pred))

MSE = np.mean((y - y_pred) ** 2)

RMSE = np.sqrt(MSE)

SS_res = np.sum((y - y_pred) ** 2)
SS_tot = np.sum((y - np.mean(y)) ** 2)

R2 = 1 - (SS_res / SS_tot)

print("\n--- Regression Evaluation Metrics ---")
print("Mean Absolute Error (MAE) =", round(MAE, 2))
print("Mean Squared Error (MSE) =", round(MSE, 2))
print("Root Mean Squared Error (RMSE) =", round(RMSE, 2))
print("R2 Score =", round(R2, 4))
