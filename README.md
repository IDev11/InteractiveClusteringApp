# InteractiveClusteringApp

This project is an interactive web application developed for clustering analysis, built using Streamlit. It enables users to upload datasets, preprocess data, apply various clustering algorithms, and visualize results through an intuitive interface. The app supports K-Means, Agglomerative Clustering (AGNES), DIANA (custom implementation), and DBSCAN, with plans to include K-Medoids in future updates. It is designed for both novice and experienced users, making clustering accessible for applications like market segmentation, user behavior analysis, and scientific research.

This project was developed as part of a Master’s in Bioinformatics at the Faculty of Computer Science, Department of Artificial Intelligence and Data Science, under the supervision of Mme. Belhadi Hiba.

## Features
- **Data Preprocessing**:
  - Upload datasets in CSV format with preview functionality.
  - Clean data by dropping irrelevant columns.
  - Handle missing values using strategies like mean, median, or mode imputation.
  - Normalize data with StandardScaler or Min-Max scaling.
- **Clustering Algorithms**:
  - K-Means: Partition-based clustering with elbow curve for optimal cluster selection.
  - AGNES: Hierarchical agglomerative clustering with dendrogram visualization.
  - DIANA: Custom divisive hierarchical clustering implementation.
  - DBSCAN: Density-based clustering with manual or auto-tuned parameters (eps, min_samples).
  - K-Medoids: Currently commented out in code but planned for future integration.
- **Visualizations**:
  - 2D and 3D cluster plots using PCA or t-SNE for dimensionality reduction.
  - Dendrograms for hierarchical methods (AGNES, DIANA).
  - Elbow curve for K-Means to determine optimal cluster count.
- **Evaluation Metrics**:
  - Inertia for K-Means (and K-Medoids when implemented).
  - Cluster distribution summaries.
  - Planned addition of Silhouette Score for enhanced evaluation.
- **Interactive Interface**:
  - User-friendly Streamlit interface with dynamic updates.
  - Flexible column selection and algorithm parameter tuning.
  - Real-time feedback and suggested parameters for DBSCAN.
- **Scalability and Flexibility**:
  - Supports diverse datasets and use cases, from small to moderately large datasets.
  - Customizable configurations for clustering and visualization.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/InteractiveClusteringApp.git
   cd InteractiveClusteringApp
   ```
2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Note: The `scikit-learn-extra` dependency is included for K-Medoids, which is currently commented out in `app.py`. You can skip it if not using K-Medoids.
4. **Run the App**:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Open the app in your browser (Streamlit typically runs at `http://localhost:8501`).
2. Upload a CSV dataset via the drag-and-drop interface.
3. Preprocess data:
   - Select columns to drop or use for clustering.
   - Handle missing values (mean, median, mode, or drop).
   - Apply normalization (StandardScaler or Min-Max).
4. Configure clustering:
   - Choose an algorithm (K-Means, AGNES, DIANA, DBSCAN).
   - Adjust parameters (e.g., number of clusters, DBSCAN’s eps and min_samples).
5. Visualize results:
   - View 2D/3D cluster plots, dendrograms, or elbow curves.
   - Explore cluster distributions and evaluation metrics.
6. Export results as images or CSV files with cluster labels.

## Project Structure
- `app.py`: Main Streamlit application for the interactive clustering interface.
- `clustering_algorithms.py`: Custom implementation of the DIANA clustering algorithm.
- `utils/`:
  - `preprocessing.py`: Functions for handling missing data and normalization.
  - `visualization.py`: Functions for generating 2D/3D plots, dendrograms, and elbow curves.
  - `dbscan_helper.py`: Helper functions for tuning DBSCAN parameters.
- `requirements.txt`: Lists project dependencies.
- `.gitignore`: Excludes unnecessary files (e.g., `__pycache__`, virtual environments, CSVs).

## Screenshots
- **Dataset Upload**: Drag-and-drop CSV upload with preview.
- **Preprocessing**: Select columns, handle missing values, and normalize data.
- **Clustering Configuration**: Choose algorithms and set parameters.
- **Visualization**: View 2D/3D clusters, dendrograms, and evaluation metrics.

(You can add screenshot images to the repository and reference them here, e.g., `![Dataset Upload](screenshots/upload.png)`.)

## Future Improvements
- Add support for additional clustering algorithms (e.g., graph-based or deep clustering).
- Implement Silhouette Score for enhanced cluster evaluation.
- Optimize performance for large datasets.
- Enhance visualizations with more interactive features (e.g., zoom, export formats).

## Contributors
- **Lamara Abdeldjalil** (212131052111)
- **Taleb Youcef** (191938012108)
- **Supervisor**: Mme. Belhadi Hiba

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
