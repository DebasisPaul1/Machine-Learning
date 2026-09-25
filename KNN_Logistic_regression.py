import pandas as pd
import numpy as np

mydata = pd.read_csv("credit_card_fraud_1000.csv")
X=np.array(mydata[["Amount","Transaction_Hour","Location_Risk","Device_Risk","Merchant_Risk"]])

y = np.array(mydata["Fraud"])

X_Test = np.array([250, 7, 47, 77, 28])

distances=[]

for i in range(len(X)):
    d = np.sqrt(np.sum((X[i] - X_Test) ** 2))
    distances.append(d)

sorted_indices = np.argsort(distances)

k = 3
nearest_indices = sorted_indices[:k]

nearest_prices = y[nearest_indices]

#predicted_price = np.mean(nearest_prices)

print ("Nearest Price:",nearest_prices)
t=[]
k=[]
for i in (nearest_prices):
	if i==0:
		t.append(i)
	else:
		k.append(i)
if (len(t)<len(k)):
	print ("FRAUD")
else:
	print ("NOT FRAUD")
	

