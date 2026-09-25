import pandas as pd
import numpy as np

mydata = pd.read_csv("credit_card_fraud_1000.csv")

X = np.array(mydata[
    ["Amount", "Transaction_Hour", "Location_Risk",
     "Device_Risk", "Merchant_Risk", "Transaction_Frequency"]
], dtype=float)

y = np.array(mydata["Fraud"], dtype=float)

y = np.where(y == 0, -1, 1)

m, n = X.shape

print("m =", m)
print("n =", n)

eta = 0.001 #learning rate
C = 1       #regularization parameter
T = 1000    #number of epochs

w = np.zeros(n)
b = 0.0

for epoch in range(T):

    for i in range(m):

        x_i = X[i]
        y_i = y[i]

        decision = np.dot(w, x_i) + b

        margin = y_i * decision

        if margin < 1:

            w = w - eta * (2 * w - C * y_i * x_i)

            b = b + eta * C * y_i

        else:

            w = w - eta * 2 * w

print("Final weights =", w)
print("Final bias =", b)

pred_val = np.dot(X, w) + b

y_pred = np.where(pred_val < 0, -1, 1)

y_pred = np.where(y_pred == -1, 0, 1)

y_actual = np.where(y == -1, 0, 1)

accuracy = np.mean(y_pred == y_actual)

print("Accuracy =", accuracy)
print("Accuracy (%) =", accuracy * 100)
