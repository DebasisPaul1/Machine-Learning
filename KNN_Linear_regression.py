import pandas as pd
import numpy as np

mydata = pd.read_csv("House Price Prediction Dataset.csv")

X = np.array(mydata[["Area", "Bedrooms", "Bathrooms", "Floors", "Year_gap"]])
y = np.array(mydata["Price"])

X_Test = np.array([3842, 2, 2, 1, 83])

distances=[]

for i in range(len(X)):
    d = np.sqrt(np.sum((X[i] - X_Test) ** 2))
    distances.append(d)

sorted_indices = np.argsort(distances)

k = 3
nearest_indices = sorted_indices[:k]

nearest_prices = y[nearest_indices]

predicted_price = np.mean(nearest_prices)

#print("Distances =", distances)
print("Nearest 3 indices =", nearest_indices)
print("Nearest 3 prices =", nearest_prices)
print("Predicted Price =", predicted_price)

