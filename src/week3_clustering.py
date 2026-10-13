"""Week 3 unsupervised-learning workflow using the built-in Iris dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.ml_workflows import MODELS_DIR, OUTPUT_DIR, ensure_directories


def load_iris_dataset() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Load scikit-learn's reliable built-in Iris data without a network request."""
    iris = load_iris()
    features = pd.DataFrame(iris.data, columns=iris.feature_names)
    targets = pd.Series(iris.target, name="true_target_id")
    return features, targets, list(iris.target_names)


def load_iris_model_artifacts(model_dir: str | Path = MODELS_DIR) -> Dict[str, Any]:
    """Load saved Iris scaler, K-Means, and PCA objects."""
    model_root = Path(model_dir)
    artifacts = {
        "scaler": joblib.load(model_root / "iris_scaler.joblib"),
        "kmeans": joblib.load(model_root / "iris_kmeans.joblib"),
        "pca": joblib.load(model_root / "iris_pca.joblib"),
    }
    return artifacts


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_iris_clustering_workflow(
    output_dir: str | Path = OUTPUT_DIR,
    model_dir: str | Path = MODELS_DIR,
    n_clusters: int = 3,
    max_k: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Fit reproducible K-Means/PCA objects and write reports and visualizations."""
    ensure_directories()
    features, true_targets, target_names = load_iris_dataset()
    if not 1 <= n_clusters <= len(features):
        raise ValueError(f"n_clusters must be between 1 and {len(features)}.")
    if max_k < n_clusters:
        raise ValueError("max_k must be greater than or equal to n_clusters.")

    output_root = Path(output_dir)
    reports_dir = output_root / "reports"
    plots_dir = output_root / "plots"
    model_root = Path(model_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    # Clustering is unsupervised; the scaler and PCA are fitted on the full
    # feature-only Iris dataset. Target labels are used only for comparison.
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    pca = PCA(n_components=2, random_state=random_state)
    pca_features = pca.fit_transform(scaled_features)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    kmeans.fit(scaled_features)
    cluster_labels = kmeans.labels_

    inertias: List[Dict[str, Any]] = []
    selected_model_inertia: float | None = None
    for k in range(1, min(max_k, len(features)) + 1):
        if k == n_clusters:
            elbow_model = kmeans
        else:
            elbow_model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            elbow_model.fit(scaled_features)
        inertias.append({"k": k, "inertia": float(elbow_model.inertia_)})
        if k == n_clusters:
            selected_model_inertia = float(elbow_model.inertia_)

    predictions_path = reports_dir / "iris_cluster_assignments.csv"
    assignments = features.copy()
    assignments["true_target_id"] = true_targets
    assignments["true_species"] = [target_names[value] for value in true_targets]
    assignments["predicted_cluster"] = cluster_labels
    assignments["pca_component_1"] = pca_features[:, 0]
    assignments["pca_component_2"] = pca_features[:, 1]
    assignments.to_csv(predictions_path, index_label="sample_index")

    scaled_centers = kmeans.cluster_centers_
    original_centers = scaler.inverse_transform(scaled_centers)
    centers_data: Dict[str, Any] = {"cluster_id": np.arange(n_clusters)}
    for column_index, feature_name in enumerate(features.columns):
        centers_data[f"{feature_name}_scaled_center"] = scaled_centers[:, column_index]
        centers_data[f"{feature_name}_original_units_center"] = original_centers[:, column_index]
    centers_path = reports_dir / "iris_cluster_centers.csv"
    pd.DataFrame(centers_data).to_csv(centers_path, index=False)

    inertia_path = reports_dir / "iris_elbow_inertia.csv"
    pd.DataFrame(inertias).to_csv(inertia_path, index=False)

    variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_ratio)
    variance_path = reports_dir / "iris_pca_explained_variance.csv"
    pd.DataFrame(
        {
            "principal_component": ["PC1", "PC2"],
            "explained_variance_ratio": variance_ratio,
            "cumulative_explained_variance": cumulative_variance,
        }
    ).to_csv(variance_path, index=False)

    cluster_species = pd.crosstab(
        pd.Series(cluster_labels, name="predicted_cluster"),
        pd.Series([target_names[value] for value in true_targets], name="true_species"),
    ).reindex(columns=target_names, fill_value=0)
    comparison_path = reports_dir / "iris_cluster_vs_species.csv"
    cluster_species.to_csv(comparison_path)

    elbow_plot_path = plots_dir / "week3_iris_elbow_method.png"
    inertia_frame = pd.DataFrame(inertias)
    plt.figure(figsize=(8, 5))
    plt.plot(inertia_frame["k"], inertia_frame["inertia"], marker="o", label="K-Means inertia")
    plt.axvline(n_clusters, color="tab:red", linestyle="--", label=f"Chosen k={n_clusters}")
    plt.title("Iris K-Means Elbow Method")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (within-cluster sum of squares)")
    plt.xticks(inertia_frame["k"])
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(elbow_plot_path, dpi=150)
    plt.close()

    cluster_plot_path = plots_dir / "week3_iris_kmeans_clusters.png"
    petal_length = "petal length (cm)"
    petal_width = "petal width (cm)"
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=assignments,
        x=petal_length,
        y=petal_width,
        hue="predicted_cluster",
        palette="deep",
        s=65,
        alpha=0.85,
    )
    plt.title("Iris K-Means Clusters (Model Uses All Four Scaled Features)")
    plt.xlabel("Petal length (cm)")
    plt.ylabel("Petal width (cm)")
    plt.legend(title="Predicted cluster")
    plt.tight_layout()
    plt.savefig(cluster_plot_path, dpi=150)
    plt.close()

    pca_plot_path = plots_dir / "week3_iris_pca_clusters.png"
    pca_plot_data = pd.DataFrame(
        {
            "PC1": pca_features[:, 0],
            "PC2": pca_features[:, 1],
            "predicted_cluster": cluster_labels,
        }
    )
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=pca_plot_data,
        x="PC1",
        y="PC2",
        hue="predicted_cluster",
        palette="deep",
        s=65,
        alpha=0.85,
    )
    plt.title("Iris K-Means Clusters Projected onto Two PCA Components")
    plt.xlabel(f"PC1 ({variance_ratio[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({variance_ratio[1]:.1%} variance)")
    plt.legend(title="Predicted cluster")
    plt.tight_layout()
    plt.savefig(pca_plot_path, dpi=150)
    plt.close()

    comparison_plot_path = plots_dir / "week3_iris_cluster_species_comparison.png"
    plt.figure(figsize=(8, 5))
    sns.heatmap(cluster_species, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.title("Iris Cluster Assignments Compared with True Species")
    plt.xlabel("True Iris species")
    plt.ylabel("Predicted cluster ID (arbitrary)")
    plt.tight_layout()
    plt.savefig(comparison_plot_path, dpi=150)
    plt.close()

    scaler_path = model_root / "iris_scaler.joblib"
    kmeans_path = model_root / "iris_kmeans.joblib"
    pca_path = model_root / "iris_pca.joblib"
    joblib.dump(scaler, scaler_path)
    joblib.dump(kmeans, kmeans_path)
    joblib.dump(pca, pca_path)

    # Exercise persistence by loading the artifacts and predicting again.
    loaded = load_iris_model_artifacts(model_root)
    loaded_predictions = loaded["kmeans"].predict(loaded["scaler"].transform(features))
    predictions_match_after_reload = bool(np.array_equal(cluster_labels, loaded_predictions))
    if not predictions_match_after_reload:
        raise RuntimeError("Reloaded K-Means model predictions differ from the fitted model.")

    unique_target_counts = {
        target_names[int(target_id)]: int(count)
        for target_id, count in true_targets.value_counts().sort_index().items()
    }
    report_path = reports_dir / "week3_iris_clustering_report.json"
    report: Dict[str, Any] = {
        "dataset": "scikit-learn built-in Iris dataset",
        "sample_count": int(features.shape[0]),
        "feature_count": int(features.shape[1]),
        "feature_names": list(features.columns),
        "target_names": target_names,
        "true_species_counts": unique_target_counts,
        "n_clusters": n_clusters,
        "random_state": random_state,
        "elbow_max_k": int(inertia_frame["k"].max()),
        "inertia_at_selected_k": selected_model_inertia,
        "cluster_centers_scaled": scaled_centers.tolist(),
        "cluster_centers_original_units": original_centers.tolist(),
        "predicted_cluster_labels": loaded_predictions.tolist(),
        "cluster_sizes": {
            str(cluster): int(count)
            for cluster, count in pd.Series(loaded_predictions).value_counts().sort_index().items()
        },
        "pca_explained_variance_ratio": variance_ratio.tolist(),
        "pca_cumulative_explained_variance": cumulative_variance.tolist(),
        "total_variance_explained_by_two_components": float(cumulative_variance[-1]),
        "cluster_ids_are_arbitrary": True,
        "predictions_match_after_joblib_reload": predictions_match_after_reload,
        "cluster_species_contingency": {
            str(cluster_id): {
                str(species): int(cluster_species.loc[cluster_id, species])
                for species in cluster_species.columns
            }
            for cluster_id in cluster_species.index
        },
        "output_files": {
            "cluster_assignments_csv": str(predictions_path),
            "cluster_centers_csv": str(centers_path),
            "elbow_inertia_csv": str(inertia_path),
            "pca_explained_variance_csv": str(variance_path),
            "cluster_species_comparison_csv": str(comparison_path),
            "elbow_plot": str(elbow_plot_path),
            "clusters_plot": str(cluster_plot_path),
            "pca_plot": str(pca_plot_path),
            "cluster_species_comparison_plot": str(comparison_plot_path),
            "report": str(report_path),
            "scaler_model": str(scaler_path),
            "kmeans_model": str(kmeans_path),
            "pca_model": str(pca_path),
        },
    }
    _save_json(report_path, report)
    return report
