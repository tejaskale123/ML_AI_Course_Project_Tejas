# Week 3: Iris Clustering with K-Means and PCA

This module is integrated into the existing course project. It loads the built-in Iris dataset from scikit-learn, so the workflow does not depend on a manual download or fabricate observations.

## Run the workflow

From PowerShell at the project root:

```powershell
$conda = "$env:USERPROFILE\miniconda3\condabin\conda.bat"
& $conda run -p .\.venv python -c "from src.week3_clustering import run_iris_clustering_workflow; print(run_iris_clustering_workflow())"
```

You can also start `main.py` and choose **6. Run Week 3 (Iris clustering and PCA)**. The menu prints the dataset shape and feature names, cluster centers, all predicted cluster IDs, PCA variance, and generated file paths.

## What the code does

1. Inspects the 150 samples, four measurements, three true target classes, feature names, and target counts.
2. Fits a `StandardScaler` to all four feature columns. Since this is an unsupervised exercise with no train/test split, it uses all feature-only observations; the true labels are not inputs to the scaler, PCA, or K-Means.
3. Fits K-Means with `k=3`, `n_init=10`, and `random_state=42`.
4. Runs the Elbow Method for `k=1` through `k=10` and saves each model's inertia for plotting.
5. Fits PCA with two components on the scaled feature data and records each component's explained variance ratio and cumulative ratio.
6. Compares cluster assignments against the true species names with a count table and heatmap.
7. Saves the fitted scaler, K-Means model, and PCA object with joblib, reloads them, and checks that the loaded K-Means model reproduces the fitted model's assignments.

K-Means cluster IDs are arbitrary labels. For example, cluster `0` does not inherently mean species class `0`; use the saved contingency table to inspect the relationship rather than comparing label numbers directly.

## Generated files

- `outputs/reports/iris_cluster_assignments.csv`: true species/target, predicted cluster ID, four original features, and two PCA coordinates per sample.
- `outputs/reports/iris_cluster_centers.csv`: cluster centers in standardized coordinates and original measurement units.
- `outputs/reports/iris_elbow_inertia.csv`: inertia for each tested K.
- `outputs/reports/iris_pca_explained_variance.csv`: per-component and cumulative explained variance.
- `outputs/reports/iris_cluster_vs_species.csv`: counts of each true species in each predicted cluster.
- `outputs/reports/week3_iris_clustering_report.json`: dataset summary, all cluster IDs, centers, variance ratios, comparison table, and output paths.
- `outputs/plots/week3_iris_elbow_method.png`
- `outputs/plots/week3_iris_kmeans_clusters.png`
- `outputs/plots/week3_iris_pca_clusters.png`
- `outputs/plots/week3_iris_cluster_species_comparison.png`
- `models/iris_scaler.joblib`, `models/iris_kmeans.joblib`, and `models/iris_pca.joblib`

All generated files stay under the existing `outputs/` and `models/` folders.

## Concepts

- **Feature scaling** puts measurements with different units/ranges on comparable scales. It matters for distance-based K-Means and PCA.
- **K-Means** assigns observations to the nearest centroid and repeatedly updates centroids to reduce within-cluster squared distances (inertia).
- **Elbow Method** plots inertia for several values of K. Inertia always tends to decline as K grows; a bend can suggest a useful K, but it is a heuristic rather than proof that one K is correct.
- **PCA** forms new, orthogonal directions that capture variance. The variance ratios describe information retained by the plotted two-dimensional projection; K-Means in this project is fit on all four scaled features, not only the PCA plot coordinates.
