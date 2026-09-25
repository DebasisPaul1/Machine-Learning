import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("driver.csv")

X = np.array(data[["mean_dist_day", "mean_over_speed_perc"]], dtype=float)

def initialize_centroids(X, K):
    indices = np.random.choice(len(X), K, replace=False)
    return X[indices]

def calculate_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid) ** 2))

def assign_clusters(X, centroids):
    clusters = []

    for point in X:
        distances = []

        for centroid in centroids:
            distance = calculate_distance(point, centroid)
            distances.append(distance)

        cluster = np.argmin(distances)
        clusters.append(cluster)

    return np.array(clusters)

def update_centroids(X, clusters, K):
    centroids = []

    for k in range(K):
        cluster_points = X[clusters == k]

        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
        else:
            centroid = X[np.random.randint(len(X))]

        centroids.append(centroid)

    return np.array(centroids)

def calculate_wcss(X, clusters, centroids):
    wcss = 0

    for i in range(len(X)):
        centroid = centroids[clusters[i]]
        distance = calculate_distance(X[i], centroid)
        wcss += distance ** 2

    return wcss

def kmeans(X, K, max_iterations=100):
    centroids = initialize_centroids(X, K)

    for iteration in range(max_iterations):
        clusters = assign_clusters(X, centroids)

        new_centroids = update_centroids(X, clusters, K)

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    wcss = calculate_wcss(X, clusters, centroids)

    return clusters, centroids, wcss

K = 3

clusters, centroids, wcss = kmeans(X, K)

print("Clusters:")
print(clusters)

print("Centroids:")
print(centroids)

print("WCSS:")
print(wcss)

plt.scatter(X[:, 0], X[:, 1], c=clusters)

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=200
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering")

plt.show()

wcss_values = []

for K in range(1, 11):
    clusters, centroids, wcss = kmeans(X, K)
    wcss_values.append(wcss)

plt.plot(range(1, 11), wcss_values, marker="o")

plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")

plt.show()
