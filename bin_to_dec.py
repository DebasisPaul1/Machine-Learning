import pandas as pd
import numpy as np

bits = 10
eta = 0.01
epochs = 100

data = pd.read_csv("10bit_binary_to_decimal (1).csv")

columns = data.columns

X = data[columns[:10]].values
y = data[columns[-1]].values

weights = np.zeros(bits)

for epoch in range(epochs):
    for j in range(len(X)):
        binary_input = X[j]
        target = y[j]

        output = 0

        for i in range(bits):
            output += weights[i] * binary_input[i]

        error = target - output

        for i in range(bits):
            weights[i] = weights[i] + eta * error * binary_input[i]

print("Final Weights:")
print(weights)

print("\nTesting:")

test = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

for binary_input in test:
    output = 0

    for i in range(bits):
        output += weights[i] * binary_input[i]

    print(binary_input, "->", output)

