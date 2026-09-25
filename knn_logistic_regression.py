import pandas as pd
import numpy as np

mydata = pd.read_csv("credit_card_fraud_1000.csv")
X = np.array(mydata[["Amount","Transaction_Hour","Location_Risk","Device_Risk","Merchant_Risk"]])


y = np.array(mydata["Fraud"])

m, n = X.shape

print("m =", m)
print("n =", n)

print("\nEnter Test Transaction Details")

amount = float(input("Enter Amount: "))
transaction_hour = float(input("Enter Transaction Hour: "))
location_risk = float(input("Enter Location Risk: "))
device_risk = float(input("Enter Device Risk: "))
merchant_risk = float(input("Enter Merchant Risk: "))

X_Test = np.array([
    amount,
    transaction_hour,
    location_risk,
    device_risk,
    merchant_risk
])

k = int(input("\nEnter K: "))

distances = np.sqrt(
    np.sum((X - X_Test) ** 2, axis=1)
)

nearest_indices = np.argsort(distances)[:k]

nearest_labels = y[nearest_indices]


fraud_count = np.sum(nearest_labels == 1)
normal_count = np.sum(nearest_labels == 0)

print("\nFraud votes =", fraud_count)
print("Normal votes =", normal_count)

if fraud_count > normal_count:
    prediction = 1
else:
    prediction = 0

print("\nPredicted Fraud =", prediction)

if prediction == 1:
    print("Transaction is FRAUD")
else:
    print("Transaction is NOT FRAUD")

