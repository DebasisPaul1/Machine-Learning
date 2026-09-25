import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("KMeans_Data.csv")

data = np.array(df[["X", "Y"]], dtype=float)


def kmeans(data, K):

    np.random.seed(42)

    random_indices = np.random.choice(len(data), K, replace=False)

    centroids = data[random_indices].copy()

    for iteration in range(100):

        clusters = [[] for _ in range(K)]

        for point in data:

            distances = []

            for centroid in centroids:

                distance = np.sqrt(
                    np.sum((point - centroid) ** 2)
                )

                distances.append(distance)

            cluster_index = np.argmin(distances)

            clusters[cluster_index].append(point)


        new_centroids = []

        for i, cluster in enumerate(clusters):

            cluster = np.array(cluster)

            if len(cluster) > 0:

                centroid = np.mean(cluster, axis=0)

            else:

                centroid = centroids[i]

            new_centroids.append(centroid)


        new_centroids = np.array(new_centroids)


        if np.allclose(centroids, new_centroids):

            break


        centroids = new_centroids


    wcss = 0

    for i in range(K):

        cluster = np.array(clusters[i])

        for point in cluster:

            wcss += np.sum(
                (point - centroids[i]) ** 2
            )


    return clusters, centroids, wcss, iteration + 1



K = 3

clusters, centroids, wcss, iterations = kmeans(data, K)


print("K-Means Clustering")

print("\nNumber of Clusters:", K)

print("Number of Iterations:", iterations)


print("\nFinal Centroids:")

for i, centroid in enumerate(centroids):

    print(
        "Cluster",
        i + 1,
        ":",
        centroid
    )


print("\nNumber of Points in Each Cluster:")

for i, cluster in enumerate(clusters):

    print(
        "Cluster",
        i + 1,
        ":",
        len(cluster)
    )


print("\nWCSS:", wcss)



plt.figure(figsize=(10, 7))


for i, cluster in enumerate(clusters):

    cluster = np.array(cluster)

    plt.scatter(
        cluster[:, 0],
        cluster[:, 1],
        s=15,
        label="Cluster " + str(i + 1)
    )


plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    s=200,
    marker="X",
    label="Centroids"
)


plt.xlabel("X")

plt.ylabel("Y")

plt.title("K-Means Clustering")

plt.legend()

plt.grid(True)

plt.show()



K_values = range(1, 11)

wcss_values = []


for K in K_values:

    clusters, centroids, wcss, iterations = kmeans(
        data,
        K
    )

    wcss_values.append(wcss)


print("\nElbow Method")

for K, wcss in zip(K_values, wcss_values):

    print(
        "K =",
        K,
        " WCSS =",
        wcss
    )



plt.figure(figsize=(10, 7))

plt.plot(
    K_values,
    wcss_values,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")

plt.ylabel("WCSS")

plt.title("Elbow Method")

plt.xticks(K_values)

plt.grid(True)

plt.show()
