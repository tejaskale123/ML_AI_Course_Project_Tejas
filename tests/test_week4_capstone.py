from __future__ import annotations

import json
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.svm import SVC

import src.ml_workflows as workflows
from src.week4_capstone import (
    MALL_FEATURES,
    _make_synthetic_mall_sample,
    load_mall_customers,
    predict_customer_clusters,
    prepare_mall_customers,
    run_customer_segmentation,
    run_week4_model_evaluation,
)


def test_mall_loader_uses_documented_reproducible_synthetic_fallback() -> None:
    first, source = load_mall_customers()
    second, second_source = load_mall_customers()

    assert source == second_source == "synthetic_fallback"
    assert len(first) == 150
    pd.testing.assert_frame_equal(first, second)
    assert set(MALL_FEATURES).issubset(first.columns)


def test_mall_preprocessing_normalizes_columns_and_removes_invalid_rows() -> None:
    raw = pd.DataFrame(
        {
            "age": [20, 30, 30, 40, None],
            "Annual Income": [20, 50, 50, 70, 75],
            "Spending Score": [30, 60, 60, 40, 80],
            "Gender": ["F", "M", "M", "F", "F"],
        }
    )
    cleaned, features, quality = prepare_mall_customers(raw)

    assert list(features.columns) == MALL_FEATURES
    assert len(cleaned) == 3
    assert quality["input_rows"] == 5
    assert quality["duplicate_rows_removed"] == 1
    assert quality["rows_missing_clustering_features_removed"] == 1
    assert not features.isna().any().any()
    assert quality["input_feature_data_types"]["Age"] == "float64"


def test_mall_preprocessing_rejects_missing_columns_and_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        prepare_mall_customers(pd.DataFrame())
    with pytest.raises(ValueError, match="missing clustering columns"):
        prepare_mall_customers(pd.DataFrame({"Age": [20]}))


def test_customer_segmentation_writes_reports_plots_and_reloadable_models(tmp_path) -> None:
    output_dir = tmp_path / "outputs"
    model_dir = tmp_path / "models"
    report = run_customer_segmentation(output_dir=output_dir, model_dir=model_dir)

    assert report["dataset_source_type"] == "synthetic_fallback"
    assert "not real customer data" in report["dataset_source"]
    assert report["input_shape"][0] == 150
    assert report["model_metrics"]["cluster_count"] == 5
    assert -1 <= report["model_metrics"]["silhouette_score"] <= 1
    assert len(report["cluster_summary"]) == 5
    assert report["hypothetical_new_customer_prediction"]["predicted_cluster_id"] in range(5)

    assignments = pd.read_csv(
        output_dir / "reports" / "week4_customer_cluster_assignments.csv"
    )
    summary = pd.read_csv(output_dir / "reports" / "week4_customer_cluster_summary.csv")
    inertia = pd.read_csv(output_dir / "reports" / "week4_customer_elbow_inertia.csv")
    assert len(assignments) == 150
    assert assignments["cluster_id"].nunique() == 5
    assert len(summary) == 5
    assert list(inertia["k"]) == list(range(1, 11))
    assert "segment_label" in summary.columns
    assert int(summary["customer_count"].sum()) == 150

    new_customer = pd.DataFrame(
        [{"Age": 32, "Annual Income (k$)": 65, "Spending Score (1-100)": 72}]
    )
    assert predict_customer_clusters(new_customer, model_dir).shape == (1,)
    with pytest.raises(ValueError, match="missing required columns"):
        predict_customer_clusters(pd.DataFrame([{"Age": 30}]), model_dir)
    loaded_scaler = joblib.load(model_dir / "week4_customer_scaler.joblib")
    loaded_model = joblib.load(model_dir / "week4_customer_kmeans.joblib")
    assert loaded_model.predict(loaded_scaler.transform(new_customer)).shape == (1,)

    assert (output_dir / "reports" / "week4_customer_data_quality.json").is_file()
    assert (
        output_dir / "reports" / "week4_customer_segmentation_report.json"
    ).is_file()
    assert (output_dir / "plots" / "week4_customer_elbow_method.png").is_file()
    assert (output_dir / "plots" / "week4_customer_segments.png").is_file()


def test_customer_segmentation_uses_local_csv_when_available(tmp_path) -> None:
    input_path = tmp_path / "Mall_Customers.csv"
    sample = _make_synthetic_mall_sample().iloc[:30]
    sample.to_csv(input_path, index=False)
    loaded, source = load_mall_customers(input_path)
    assert source == "local_mall_customers_csv"
    pd.testing.assert_frame_equal(loaded, sample)

    report = run_customer_segmentation(
        dataset_path=input_path,
        output_dir=tmp_path / "outputs",
        model_dir=tmp_path / "models",
        n_clusters=3,
        max_k=5,
    )
    assert report["dataset_source_type"] == "local_mall_customers_csv"
    assert report["input_shape"][0] == 30


@pytest.mark.parametrize(
    ("n_clusters", "max_k"),
    [(1, 5), (150, 150), (151, 151), (5, 4)],
)
def test_customer_segmentation_rejects_invalid_parameters(
    tmp_path, n_clusters, max_k
) -> None:
    with pytest.raises(ValueError):
        run_customer_segmentation(
            output_dir=tmp_path / "outputs",
            model_dir=tmp_path / "models",
            n_clusters=n_clusters,
            max_k=max_k,
        )


def test_svm_evaluation_runs_cv_grid_search_saves_loadable_model(tmp_path) -> None:
    report = run_week4_model_evaluation(
        output_dir=tmp_path / "outputs",
        model_dir=tmp_path / "models",
    )

    assert report["sample_count"] == 569
    assert len(report["cross_validation"]["fold_accuracies"]) == 5
    assert 0 <= report["cross_validation"]["mean_accuracy"] <= 1
    assert set(report["grid_search"]["best_parameters"]) == {
        "svm__C",
        "svm__kernel",
        "svm__gamma",
    }
    assert 0 <= report["held_out_metrics"]["accuracy"] <= 1
    assert report["held_out_metrics"]["precision_recall_positive_class"] == "benign"
    assert report["reload_predictions_match"] is True
    assert len(report["new_sample_predictions"]) == 3
    matrix = np.asarray(report["held_out_metrics"]["confusion_matrix"])
    assert matrix.shape == (2, 2)
    assert int(matrix.sum()) == 114

    model = joblib.load(tmp_path / "models" / "week4_svm_best_model.joblib")
    assert isinstance(model.named_steps["svm"], SVC)
    assert (tmp_path / "outputs" / "reports" / "week4_svm_cross_validation_scores.csv").is_file()
    assert (tmp_path / "outputs" / "reports" / "week4_svm_grid_search_results.csv").is_file()
    assert (tmp_path / "outputs" / "reports" / "week4_svm_predictions.csv").is_file()
    assert (tmp_path / "outputs" / "reports" / "week4_svm_evaluation_report.json").is_file()
    assert (tmp_path / "outputs" / "plots" / "week4_svm_confusion_matrix.png").is_file()
    assert (tmp_path / "outputs" / "plots" / "week4_svm_cross_validation_scores.png").is_file()

    saved = json.loads(
        (tmp_path / "outputs" / "reports" / "week4_svm_evaluation_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["reload_predictions_match"] is True


def test_menu_dispatches_week4_without_removing_exit(capsys) -> None:
    stub = {
        "customer_segmentation": {
            "dataset_source": "synthetic fallback",
            "model_metrics": {"inertia": 2.0},
            "cluster_summary": [],
            "output_files": {},
        },
        "model_evaluation": {
            "cross_validation": {"mean_accuracy": 0.9},
            "grid_search": {"best_parameters": {}},
            "held_out_metrics": {"accuracy": 0.9},
            "output_files": {},
        },
    }
    with patch("builtins.input", side_effect=["7", "4"]), patch(
        "src.week4_capstone.run_week4_workflows", return_value=stub
    ) as run_week4:
        workflows.terminal_menu()

    run_week4.assert_called_once_with()
    output = capsys.readouterr().out
    assert "Run Week 4" in output
    assert "synthetic fallback" in output
    assert "Goodbye." in output
