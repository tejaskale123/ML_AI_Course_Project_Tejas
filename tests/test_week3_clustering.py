from __future__ import annotations

import json
from unittest.mock import patch

import joblib
import pandas as pd
import pytest

import src.ml_workflows as workflows
from src.week3_clustering import (
    load_iris_dataset,
    load_iris_model_artifacts,
    run_iris_clustering_workflow,
)


def test_builtin_iris_dataset_has_expected_features_and_species() -> None:
    features, targets, target_names = load_iris_dataset()

    assert features.shape == (150, 4)
    assert list(features.columns) == [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]
    assert targets.nunique() == 3
    assert target_names == ["setosa", "versicolor", "virginica"]


def test_week3_workflow_generates_reports_plots_and_reloadable_models(tmp_path) -> None:
    output_dir = tmp_path / "outputs"
    model_dir = tmp_path / "models"
    report = run_iris_clustering_workflow(output_dir, model_dir)

    assert report["sample_count"] == 150
    assert report["feature_count"] == 4
    assert report["n_clusters"] == 3
    assert report["cluster_ids_are_arbitrary"] is True
    assert report["predictions_match_after_joblib_reload"] is True
    assert len(report["predicted_cluster_labels"]) == 150
    assert len(report["cluster_centers_scaled"]) == 3
    assert len(report["cluster_centers_original_units"]) == 3
    assert len(report["pca_explained_variance_ratio"]) == 2
    assert report["pca_cumulative_explained_variance"][-1] == pytest.approx(
        sum(report["pca_explained_variance_ratio"])
    )

    assignments = pd.read_csv(output_dir / "reports" / "iris_cluster_assignments.csv")
    centers = pd.read_csv(output_dir / "reports" / "iris_cluster_centers.csv")
    inertia = pd.read_csv(output_dir / "reports" / "iris_elbow_inertia.csv")
    variance = pd.read_csv(output_dir / "reports" / "iris_pca_explained_variance.csv")
    comparison = pd.read_csv(output_dir / "reports" / "iris_cluster_vs_species.csv")

    assert len(assignments) == 150
    assert assignments["predicted_cluster"].nunique() == 3
    assert len(centers) == 3
    assert list(inertia["k"]) == list(range(1, 11))
    assert len(variance) == 2
    assert set(comparison.columns) == {
        "predicted_cluster",
        "setosa",
        "versicolor",
        "virginica",
    }
    assert int(comparison[["setosa", "versicolor", "virginica"]].to_numpy().sum()) == 150

    artifact_paths = [
        model_dir / "iris_scaler.joblib",
        model_dir / "iris_kmeans.joblib",
        model_dir / "iris_pca.joblib",
    ]
    artifacts = load_iris_model_artifacts(model_dir)
    assert set(artifacts) == {"scaler", "kmeans", "pca"}
    assert all(path.is_file() for path in artifact_paths)
    assert all(path.stat().st_size > 0 for path in artifact_paths)
    assert len(artifacts["kmeans"].predict(artifacts["scaler"].transform(load_iris_dataset()[0]))) == 150
    assert joblib.load(model_dir / "iris_pca.joblib").n_components_ == 2

    plot_names = [
        "week3_iris_elbow_method.png",
        "week3_iris_kmeans_clusters.png",
        "week3_iris_pca_clusters.png",
        "week3_iris_cluster_species_comparison.png",
    ]
    assert all((output_dir / "plots" / name).is_file() for name in plot_names)
    saved_report = json.loads(
        (output_dir / "reports" / "week3_iris_clustering_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved_report["predictions_match_after_joblib_reload"] is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_clusters": 0}, "n_clusters"),
        ({"n_clusters": 151}, "n_clusters"),
        ({"n_clusters": 3, "max_k": 2}, "max_k"),
    ],
)
def test_week3_workflow_rejects_invalid_cluster_parameters(
    tmp_path, kwargs, message
) -> None:
    with pytest.raises(ValueError, match=message):
        run_iris_clustering_workflow(
            output_dir=tmp_path / "outputs",
            model_dir=tmp_path / "models",
            **kwargs,
        )


def test_menu_dispatches_week3_and_preserves_week1_exit(capsys) -> None:
    with patch("builtins.input", side_effect=["6", "4"]), patch(
        "src.week3_clustering.run_iris_clustering_workflow",
        return_value={
            "sample_count": 150,
            "feature_count": 4,
            "feature_names": ["petal length", "petal width"],
            "target_names": ["setosa", "versicolor", "virginica"],
            "cluster_centers_scaled": [],
            "cluster_centers_original_units": [],
            "predicted_cluster_labels": [0, 1, 2],
            "pca_explained_variance_ratio": [0.7, 0.2],
            "pca_cumulative_explained_variance": [0.7, 0.9],
            "predictions_match_after_joblib_reload": True,
            "output_files": {},
        },
    ) as run_week3:
        workflows.terminal_menu()

    run_week3.assert_called_once_with()
    output = capsys.readouterr().out
    assert "Run Week 3" in output
    assert "predicted_cluster_labels" in output
    assert "Goodbye." in output
