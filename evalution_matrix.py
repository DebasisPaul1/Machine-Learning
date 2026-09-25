import pandas as pd
import numpy as np

mydata = pd.read_csv("credit_card_fraud_1000.csv")

X = np.array(mydata[
    ["Amount", "Transaction_Hour", "Location_Risk",
     "Device_Risk", "Merchant_Risk", "Transaction_Frequency"]
], dtype=float)

y = np.array(mydata["Fraud"], dtype=int)

m, n = X.shape

print("m =", m)
print("n =", n)

np.random.seed(42)

indices = np.random.permutation(m)

train_size = int(0.8 * m)

train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train = X[train_indices]
X_test = X[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

std[std == 0] = 1

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

W = np.zeros(n)
b = 0

alpha = 0.01
epochs = 10000

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for i in range(epochs):

    z = np.dot(X_train, W) + b

    y_pred = sigmoid(z)

    error = y_pred - y_train

    dW = (1 / len(X_train)) * np.dot(X_train.T, error)

    db = (1 / len(X_train)) * np.sum(error)

    W = W - alpha * dW

    b = b - alpha * db

print("\nW =", W)

print("\nb =", b)

X_Test = np.array([
    200,
    23,
    80,
    70,
    60,
    8
], dtype=float)

X_Test = (X_Test - mean) / std

z = np.dot(X_Test, W) + b

Y_pred = sigmoid(z)

print("\nFraud Probability =", Y_pred)

if Y_pred >= 0.5:
    print("1")
    print("Transaction is FRAUD")
else:
    print("0")
    print("Transaction is NOT FRAUD")

y_test_score = sigmoid(np.dot(X_test, W) + b)

y_test_class = (y_test_score >= 0.5).astype(int)

result = pd.DataFrame({
    "Actual_Fraud": y_test,
    "Predicted_Score": y_test_score,
    "Actual_Class": y_test,
    "Predicted_Class": y_test_class
})

print("\n--- Actual vs Predicted ---")
print(result)

TP = 0
TN = 0
FP = 0
FN = 0

for actual, predicted in zip(y_test, y_test_class):

    if actual == 1 and predicted == 1:
        TP += 1

    elif actual == 0 and predicted == 0:
        TN += 1

    elif actual == 0 and predicted == 1:
        FP += 1

    elif actual == 1 and predicted == 0:
        FN += 1

print("\nTP =", TP)
print("TN =", TN)
print("FP =", FP)
print("FN =", FN)

accuracy = (TP + TN) / (TP + TN + FP + FN)

precision = TP / (TP + FP) if TP + FP != 0 else 0

recall = TP / (TP + FN) if TP + FN != 0 else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall != 0
    else 0
)

specificity = (
    TN / (TN + FP)
    if TN + FP != 0
    else 0
)

print("\n--- Evaluation Metrics ---")
print("Accuracy    =", round(accuracy, 2))
print("Precision   =", round(precision, 2))
print("Recall      =", round(recall, 2))
print("F1 Score    =", round(f1, 2))
print("Specificity =", round(specificity, 2))

X_New = np.array([
    250,
    22,
    85,
    90,
    80,
    8
], dtype=float)

X_New = (X_New - mean) / std

fraud_score = sigmoid(np.dot(X_New, W) + b)

print("\nPredicted Fraud Probability =", round(fraud_score, 4))

if fraud_score >= 0.5:
    print("Transaction is FRAUD")
else:
    print("Transaction is NOT FRAUD")
