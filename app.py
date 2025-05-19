# app.py
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
#from sklearn_extra.cluster import KMedoids
from scipy.cluster.hierarchy import linkage

from clustering_algorithms import diana
from utils.preprocessing import handle_missing_data, normalize_data
from utils.visualization import plot_2d_clusters, plot_3d_clusters, plot_elbow_curve, plot_dendrogram, show_metrics
from utils.dbscan_helper import find_best_dbscan_params


st.title("Interactive Clustering App")

# Upload dataset
file = st.file_uploader("Upload your dataset", type=['csv'])
if file:
    df = pd.read_csv(file)
    st.write("Dataset Preview:", df.head())

    # Drop columns
    drop_cols = st.multiselect("Select columns to drop", df.columns.tolist())
    df.drop(columns=drop_cols, inplace=True)

    # Column selection for preprocessing
    selected_column = st.selectbox("Select a column to preprocess", df.columns.tolist())
    if selected_column:
        missing_count = df[selected_column].isnull().sum()
        st.write(f"Missing values in '{selected_column}': {missing_count}")

        # Missing value handling
        strategy = st.selectbox(f"Handle missing values in '{selected_column}'", ["none", "mean", "median", "most_frequent"])
        df = handle_missing_data(df, {selected_column: strategy})

        # Normalization
        norm = st.selectbox(f"Normalization for '{selected_column}'", ["none", "standard", "minmax"])
        df = normalize_data(df, {selected_column: norm})

        st.write("Updated Dataset Preview:", df.head())

    # Final dataset selection
    st.subheader("Clustering Configuration")
    # Checkbox to select all columns
    select_all = st.checkbox("Select all columns for clustering", value=False)
    
    # If "Select all columns" is checked, select all columns in the dataset
    if select_all:
        columns = df.columns.tolist()  # Select all columns
    else:
        # Let the user select specific columns for clustering
        columns = st.multiselect("Select columns for clustering", df.columns.tolist())
        
    if columns:
        selected_df = df[columns]
        algorithm = st.selectbox("Choose clustering algorithm", ["kmeans", "kmedoids", "agnes", "diana", "dbscan"])
        X = selected_df.values

        if algorithm == "kmeans":
            k = st.slider("Number of clusters", 2, 10, 3)
            model = KMeans(n_clusters=k)
            labels = model.fit_predict(X)
            centroids = model.cluster_centers_
        #elif algorithm == "kmedoids":
         #   k = st.slider("Number of clusters", 2, 10, 3)
            #model = KMedoids(n_clusters=k)
            #labels = model.fit_predict(X)
            #centroids = model.cluster_centers_
        elif algorithm == "agnes":
            k = st.slider("Number of clusters", 2, 10, 3)
            model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(X)
            Z = linkage(X, method='ward')
            plot_dendrogram(Z)
            centroids = None
        elif algorithm == "diana":
            n_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=3)
            labels, inertia = diana(X, n_clusters=n_clusters)
            st.session_state["labels"] = labels
            st.success(f"DIANA clustering done with inertia = {inertia:.2f}")
        elif algorithm == "dbscan":
            dbscan_mode = st.radio("Choose DBSCAN mode:", ["Manual", "Auto (suggested eps & min_samples)"])
            if dbscan_mode == "Manual":
                eps = st.slider("eps", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
                min_samples = st.slider("min_samples", min_value=2, max_value=20, value=5)
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(X)

            else:
                target_clusters = st.slider("Target number of clusters", min_value=2, max_value=10, value=3)
                st.info("Finding best eps and min_samples...")
                eps, min_samples, labels = find_best_dbscan_params(X, target_clusters=target_clusters)
                if eps is None:
                    st.error("No suitable eps and min_samples found.")
                    st.stop()
                else:
                    st.success(f"Best eps: {eps:.2f}, min_samples: {min_samples}")

        st.subheader("Cluster Visualization")
        if algorithm in ["kmeans", "kmedoids"]:
            plot_elbow_curve(X, max_k=10)
            st.write("Elbow curve for optimal k")
            st.write(f"Centroids: {centroids}")
        else:
            if algorithm == "agnes":
                st.write("Dendrogram for Agglomerative Clustering")
            elif algorithm == "diana":
                # Perform hierarchical clustering (Divisive Hierarchical Clustering)
                Z = linkage(X, method='ward')  # Use appropriate linkage method (e.g., 'ward', 'single', 'complete')
                
                # Plot the dendrogram
                plot_dendrogram(Z, truncate_mode='level', p=3, leaf_rotation=90., leaf_font_size=12., show_contracted=True)

        if algorithm in ["diana", "dbscan"]:
            plot_2d_clusters(X, labels, centroids=None)
            if algorithm == "dbscan":
                show_metrics(X, labels)
        else:
            plot_2d_clusters(X, labels, centroids)
            plot_3d_clusters(X, labels)
        

        st.subheader("Evaluation Metrics")
        show_metrics(X, labels)

        # Additional metrics
        st.write("\n### Cluster Summary")
        st.write(pd.DataFrame({"Cluster": labels}).value_counts().rename("Count").reset_index())