# clustering_algorithms.py
from scipy.spatial.distance import pdist, squareform
import numpy as np


def average_dissimilarity(point_idx, group, distance_matrix):
    if len(group) == 0:
        return 0
    return np.mean([distance_matrix[point_idx][i] for i in group])

def calculate_inertia(X, labels):
    n_clusters = len(np.unique(labels))
    inertia = 0.0
    for idx in range(n_clusters):
        cluster_points = X[labels == idx]
        cluster_center = np.mean(cluster_points, axis=0)
        inertia += np.sum((cluster_points - cluster_center) ** 2)
    return inertia

def diana(X, n_clusters=2):
    distance_matrix = squareform(pdist(X, metric='euclidean'))
    n = len(X)
    clusters = [list(range(n))]

    while len(clusters) < n_clusters:
        diameters = []
        for cluster in clusters:
            if len(cluster) < 2:
                diameters.append(0)
            else:
                dists = [distance_matrix[i][j] for i in cluster for j in cluster if i != j]
                diameters.append(max(dists))
        idx_to_split = np.argmax(diameters)
        cluster_to_split = clusters.pop(idx_to_split)

        avg_dissims = [
            average_dissimilarity(i, [j for j in cluster_to_split if j != i], distance_matrix)
            for i in cluster_to_split
        ]
        splinter = [cluster_to_split[np.argmax(avg_dissims)]]
        remainder = [i for i in cluster_to_split if i not in splinter]

        moved = True
        while moved:
            moved = False
            for i in remainder[:]:
                d_to_splinter = average_dissimilarity(i, splinter, distance_matrix)
                d_to_remainder = average_dissimilarity(i, [j for j in remainder if j != i], distance_matrix)
                if d_to_splinter < d_to_remainder:
                    splinter.append(i)
                    remainder.remove(i)
                    moved = True

        clusters.append(splinter)
        clusters.append(remainder)

    labels = np.zeros(n, dtype=int)
    for idx, cluster in enumerate(clusters):
        for i in cluster:
            labels[i] = idx

    inertia = calculate_inertia(X, labels)
    return labels, inertia