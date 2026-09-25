import pandas as pd
import numpy as np

data = pd.read_csv("House Price Prediction Dataset.csv")



X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values



def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))



def knn_regression(X, y, test_point, k):

    distances = []

    for i in range(len(X)):
        distance = euclidean_distance(X[i], test_point)
        distances.append((distance, y[i]))

    
    distances.sort(key=lambda x: x[0])

    
    nearest = distances[:k]

   
    values = [item[1] for item in nearest]

    
    prediction = np.mean(values)

    return prediction



print("Enter values for the test data:")

test_point = []

for feature in data.columns[:-1]:
    value = float(input(f"{feature}: "))
    test_point.append(value)

test_point = np.array(test_point)


k = int(input("Enter value of K: "))


prediction = knn_regression(X, y, test_point, k)

print("\nPredicted Value:", prediction)
