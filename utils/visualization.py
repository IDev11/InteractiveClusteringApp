# utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from mpl_toolkits.mplot3d import Axes3D


def plot_2d_clusters(X, labels, centroids=None):
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=labels, palette='Set2', ax=ax, s=70, edgecolor='black')

    if centroids is not None:
        centroids_reduced = pca.transform(centroids)
        ax.scatter(centroids_reduced[:, 0], centroids_reduced[:, 1], marker='X', color='red', s=150, label='Centroids')
        ax.legend()

    ax.set_title("Cluster visualization (2D PCA)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(True)
    st.pyplot(fig)

def plot_clusters(X, labels):
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='Set2', edgecolor='black')
    ax.set_title("Clusters (2D Projection)")
    st.pyplot(fig)

def plot_3d_clusters(X, labels):
    pca = PCA(n_components=3)
    X_reduced = pca.fit_transform(X)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=labels, cmap='tab10', s=50)
    plt.title("3D Cluster Plot")
    st.pyplot(fig)


def plot_elbow_curve(data, max_k=10):
    from sklearn.cluster import KMeans
    distortions = []
    K = range(1, max_k+1)
    for k in K:
        model = KMeans(n_clusters=k)
        model.fit(data)
        distortions.append(model.inertia_)

    plt.figure(figsize=(6, 4))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('Elbow Method For Optimal k')
    st.pyplot(plt.gcf())
    plt.clf()


def plot_dendrogram(model, **kwargs):
    from scipy.cluster.hierarchy import dendrogram
    plt.figure(figsize=(10, 6))
    dendrogram(model, **kwargs)
    plt.title('Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    st.pyplot(plt.gcf())
    plt.clf()


def show_metrics(X, labels):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    st.write(f"Number of clusters (excluding noise): {n_clusters}")

    if n_clusters < 2:
        st.warning("Cannot compute metrics: need at least 2 clusters.")
        return

    try:
        silhouette = silhouette_score(X, labels)
        db_index = davies_bouldin_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)

        st.write(f"Silhouette Score: {silhouette:.3f}")
        st.write(f"Davies–Bouldin Index: {db_index:.3f}")
        st.write(f"Calinski–Harabasz Score: {calinski:.3f}")
    except Exception as e:
        st.error(f"Error while computing metrics: {e}")