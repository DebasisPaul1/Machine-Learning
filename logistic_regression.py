import pandas as pd
import numpy as np

mydata = pd.read_csv("credit_card_fraud_1000.csv")
X = np.array(mydata[
    ["Amount","Transaction_Hour","Location_Risk",
        "Device_Risk","Merchant_Risk","Transaction_Frequency"
    ]
])

y = np.array(mydata["Fraud"])

m, n = X.shape

print("m =", m)
print("n =", n)

W = np.zeros(n)
b = 0

alpha = 0.0001

epochs = 10000

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for i in range(epochs):

    z = np.dot(X, W) + b
    y_pred = sigmoid(z)

    error = y_pred - y
    
    dW = (1 / m) * np.dot(X.T, error)

    db = (1 / m) * np.sum(error)
    W = W - alpha * dW
    b = b - alpha * db
    
print("\nW =", W)

print("\nb =", b)

X_Test = np.array([200,23,80,70,60,   
    8      
])

z = np.dot(X_Test, W) + b
y_pred = sigmoid(z)


print("\nFraud Probability =", y_pred)

if y_pred >= 0.5:
    print("1")
    print("Transaction is FRAUD")
else:
    print("0")
    print("Transaction is NOT FRAUD")

