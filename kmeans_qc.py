import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

Load data
data = np.loadtxt('data.csv', delimiter=',')

Perform K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(data)

Plot the clusters
plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='*', s=200)
plt.show()

Print the cluster labels
print(kmeans.labels_)

Print the cluster centroids
print(kmeans.cluster_centers_)
feature1,feature2,feature3
1,2,3
4,5,6
7,8,9
10,11,12
