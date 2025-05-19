from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np

def find_best_dbscan_params(X, target_clusters=3, eps_range=(0.1, 5), min_samples_range=(2, 10)):
    best_eps, best_min_samples = None, None
    best_score = -1
    best_labels = None

    for eps in np.arange(*eps_range, 0.1):
        for min_samples in range(*min_samples_range):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters == target_clusters:
                if len(set(labels)) > 1:
                    try:
                        score = silhouette_score(X, labels)
                        if score > best_score:
                            best_score = score
                            best_eps = eps
                            best_min_samples = min_samples
                            best_labels = labels
                    except:
                        continue

    return best_eps, best_min_samples, best_labels
