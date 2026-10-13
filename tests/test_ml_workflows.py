from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest

import src.ml_workflows as workflows
from src.ml_workflows import (
    PROJECT_ROOT,
    evaluate_classification,
    prepare_house_data,
    prepare_titanic_data,
    run_house_price_workflow,
    run_titanic_workflow,
    run_week2_titanic_comparison,
    safe_load_csv,
)


def test_safe_load_csv_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        safe_load_csv("does_not_exist.csv")


def test_prepare_house_data_handles_missing_price() -> None:
    df = pd.DataFrame({"area_sqft": [500, 650, 800], "price": [85000, 112000, None]})
    X, y = prepare_house_data(df)
    assert list(X.columns) == ["area_sqft"]
    assert y.isnull().sum() == 0


def test_prepare_titanic_data_handles_missing_and_encodes_features() -> None:
    df = pd.DataFrame(
        {
            "survived": [1, 0, 1, 0],
            "pclass": [1, 2, 3, 1],
            "sex": ["male", "female", "male", "female"],
            "age": [22, None, 28, 35],
            "sibsp": [1, 0, 1, 0],
            "parch": [0, 1, 0, 0],
            "fare": [10.0, None, 25.0, 40.0],
            "embarked": ["S", "C", None, "Q"],
        }
    )
    X, y = prepare_titanic_data(df)
    assert list(X.columns) == ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    assert X["age"].isna().sum() == 1
    assert X["embarked"].isna().sum() == 1
    assert y.tolist() == [1, 0, 1, 0]


def test_prepare_titanic_data_rejects_missing_target_and_empty_data() -> None:
    with pytest.raises(ValueError, match="empty"):
        prepare_titanic_data(pd.DataFrame())
    frame = pd.DataFrame(
        {
            "survived": [None],
            "pclass": [1],
            "sex": ["female"],
            "age": [20],
            "sibsp": [0],
            "parch": [0],
            "fare": [10],
            "embarked": ["S"],
        }
    )
    with pytest.raises(ValueError, match="missing values"):
        prepare_titanic_data(frame)


def test_house_price_workflow_saves_outputs(tmp_path) -> None:
    csv_path = tmp_path / "house.csv"
    pd.DataFrame({
        "area_sqft": [500, 650, 800, 1000, 1200, 1400],
        "price": [85000, 112000, 135000, 170000, 202000, 240000],
    }).to_csv(csv_path, index=False)

    report = run_house_price_workflow(csv_path, tmp_path, tmp_path / "processed")
    assert "metrics" in report
    assert report["metrics"]["r2"] > 0.5
    assert (tmp_path / "plots" / "house_price_predictions.png").exists()
    assert (tmp_path / "reports" / "house_price_report.json").exists()
    assert (tmp_path / "reports" / "house_price_predictions.csv").exists()
    assert (tmp_path / "processed" / "house_prices_model_data.csv").exists()
    assert (PROJECT_ROOT / "models" / "house_price_linear_regression.joblib").exists()


def test_titanic_workflow_saves_outputs(tmp_path) -> None:
    csv_path = tmp_path / "titanic.csv"
    pd.DataFrame(
        {
            "survived": [1, 0, 1, 0, 1, 0, 1, 0],
            "pclass": [1, 1, 2, 3, 1, 2, 3, 1],
            "sex": ["male", "female", "male", "female", "female", "male", "female", "male"],
            "age": [22, 38, 26, None, 35, 54, 2, 40],
            "sibsp": [1, 0, 0, 1, 0, 0, 3, 0],
            "parch": [0, 1, 0, 2, 0, 1, 1, 0],
            "fare": [7.25, 71.28, 7.92, None, 53.10, 51.86, 21.08, 8.05],
            "embarked": ["S", "C", "S", "Q", "S", "S", "Q", "C"],
        }
    ).to_csv(csv_path, index=False)

    report = run_titanic_workflow(csv_path, tmp_path, tmp_path / "processed")
    assert "metrics" in report
    assert report["metrics"]["accuracy"] >= 0.0
    assert (tmp_path / "plots" / "titanic_age_distribution.png").exists()
    assert (tmp_path / "reports" / "titanic_model_report.json").exists()
    assert pd.read_csv(tmp_path / "processed" / "titanic_cleaned.csv").shape[0] == 8


def test_regression_and_classification_metrics_are_numeric() -> None:
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 0, 0]
    regression = evaluate_classification(y_true, y_pred)
    assert all(isinstance(value, float) for value in regression.values())
    assert set(regression) == {"accuracy", "precision", "recall", "f1"}


def test_week2_titanic_workflow_saves_comparison_predictions_and_models(tmp_path) -> None:
    dataset_path = PROJECT_ROOT / "data" / "raw" / "titanic.csv"
    if not dataset_path.exists():
        pytest.skip("Genuine Titanic dataset is not present.")

    report = run_week2_titanic_comparison(
        dataset_path,
        output_dir=tmp_path / "outputs",
        model_dir=tmp_path / "models",
    )

    assert report["dataset_rows"] == 891
    assert report["decision_tree_max_depth_selected_by_validation"] in {
        1, 2, 3, 4, 5, 8, None
    }
    assert {row["model"] for row in report["model_metrics"]} == {
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "K-Nearest Neighbors",
    }
    assert all(0.0 <= row["accuracy"] <= 1.0 for row in report["model_metrics"])
    assert len(pd.read_csv(tmp_path / "outputs" / "reports" / "titanic_model_comparison.csv")) == 4
    assert len(pd.read_csv(tmp_path / "outputs" / "reports" / "titanic_predictions.csv")) == 179
    assert len(pd.read_csv(tmp_path / "outputs" / "reports" / "titanic_tree_depth_validation.csv")) == 7
    assert len(list((tmp_path / "models").glob("titanic_*.joblib"))) == 4
    assert (tmp_path / "outputs" / "plots" / "titanic_model_comparison.png").exists()
    assert (tmp_path / "outputs" / "plots" / "titanic_tree_depth_validation.png").exists()
    assert (tmp_path / "outputs" / "plots" / "titanic_confusion_matrices.png").exists()


def test_menu_preserves_week1_exit_and_dispatches_week2(capsys) -> None:
    with patch("builtins.input", side_effect=["5", "4"]), patch.object(
        workflows, "run_house_price_workflow", return_value={"metrics": {"r2": 0.5}}
    ) as house_run, patch.object(
        workflows,
        "run_week2_titanic_comparison",
        return_value={"model_metrics": [{"model": "test", "accuracy": 1.0}]},
    ) as titanic_run:
        workflows.terminal_menu()
    house_run.assert_called_once_with()
    titanic_run.assert_called_once_with()
    assert "Week 2" in capsys.readouterr().out

    with patch("builtins.input", side_effect=["4"]):
        workflows.terminal_menu()
    assert "Goodbye." in capsys.readouterr().out
