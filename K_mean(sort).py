import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("driver.csv")

x = data[["mean_dist_day", "mean_over_speed_perc"]].values

k = 2
centroids = x[:k].copy()

for i in range(5):

    clusters = [[] for j in range(k)]

    for point in x:
        d = []

        for c in centroids:
            d.append(np.sqrt(np.sum((point - c) ** 2)))

        cluster = np.argmin(d)
        clusters[cluster].append(point)

    new_centroids = []

    for j in range(k):
        new_centroids.append(np.mean(clusters[j], axis=0))

    new_centroids = np.array(new_centroids)

    if np.allclose(centroids, new_centroids):
        break

    centroids = new_centroids

print("Centroids:")
print(centroids)

for i in range(k):
    print("Cluster", i + 1, ":", len(clusters[i]), "drivers")


# Graph

for i in range(k):
    cluster = np.array(clusters[i])
    plt.scatter(cluster[:, 0], cluster[:, 1])

plt.scatter(centroids[:, 0], centroids[:, 1],
            marker="X", s=200)

plt.xlabel("Mean Distance Per Day")
plt.ylabel("Mean Over Speed Percentage")
plt.title("K-Means Clustering")
plt.show()
