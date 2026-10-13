from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.ml_workflows import (
    evaluate_classification,
    prepare_house_data,
    prepare_titanic_data,
    run_house_price_workflow,
    run_titanic_workflow,
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
    assert set(X.columns) >= {"pclass", "age", "sibsp", "parch", "fare", "sex_female", "sex_male", "embarked_C", "embarked_Q", "embarked_S"}
    assert X.isna().sum().sum() == 0
    assert y.tolist() == [1, 0, 1, 0]


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
