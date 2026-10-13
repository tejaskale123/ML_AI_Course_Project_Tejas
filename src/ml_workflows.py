from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"
MODELS_DIR = PROJECT_ROOT / "models"


def ensure_directories() -> None:
    for directory in [RAW_DIR, PROCESSED_DIR, PLOTS_DIR, REPORTS_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def safe_load_csv(path: str | Path, required_columns: Iterable[str] | None = None) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if required_columns:
        missing = [column for column in required_columns if column not in df.columns]
        if missing:
            raise ValueError(f"Required columns missing: {missing}")
    return df


def prepare_house_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if "price" not in df.columns:
        raise ValueError("House price dataset must include a 'price' column.")
    working = df.copy()
    for column in working.columns:
        if working[column].dtype == object:
            working[column] = working[column].astype(str).str.strip()
    working = working.dropna(subset=["price"]).copy()
    y = working["price"].astype(float)
    X = working.drop(columns=["price"])
    return X, y


def prepare_titanic_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if df.empty:
        raise ValueError("Titanic dataset is empty.")
    required_columns = {"survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Titanic dataset missing required columns: {sorted(missing_columns)}")
    working = df.copy()
    if working["survived"].isna().any():
        raise ValueError("Titanic target column 'survived' contains missing values.")
    working["survived"] = working["survived"].astype(int)
    if not working["survived"].isin([0, 1]).all():
        raise ValueError("Titanic target column 'survived' must contain only 0 and 1.")
    features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    return working[features].copy(), working["survived"]


def _build_titanic_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    numeric_features = ["pclass", "age", "sibsp", "parch", "fare"]
    categorical_features = ["sex", "embarked"]
    numeric_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_features),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(y_true, y_pred),
    }
    return {key: float(value) for key, value in metrics.items()}


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    return {key: float(value) for key, value in metrics.items()}


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_titanic_dataset(dataset_path: str | Path | None = None) -> pd.DataFrame:
    ensure_directories()
    raw_dataset_path = Path(dataset_path) if dataset_path else RAW_DIR / "titanic.csv"
    if raw_dataset_path.exists():
        return safe_load_csv(
            raw_dataset_path,
            required_columns=["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"],
        )
    if dataset_path is not None:
        raise FileNotFoundError(f"Titanic CSV file not found: {raw_dataset_path}")
    try:
        df = sns.load_dataset("titanic")
    except Exception:
        raise FileNotFoundError(
            "Titanic dataset is unavailable. Place it at data/raw/titanic.csv or ensure seaborn can download it."
        )
    required = {"survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Downloaded Titanic dataset missing required columns: {sorted(missing)}")
    df.to_csv(raw_dataset_path, index=False)
    return df


def run_house_price_workflow(
    data_path: str | Path = RAW_DIR / "house_prices_sample.csv",
    output_dir: str | Path = OUTPUT_DIR,
    processed_dir: str | Path = PROCESSED_DIR,
) -> Dict[str, Any]:
    ensure_directories()
    csv_path = Path(data_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "plots").mkdir(parents=True, exist_ok=True)
    (output_root / "reports").mkdir(parents=True, exist_ok=True)

    df = safe_load_csv(csv_path, required_columns=["area_sqft", "price"])
    X, y = prepare_house_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = list(X_train.select_dtypes(include=[np.number]).columns)
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = evaluate_regression(np.asarray(y_test), np.asarray(predictions))

    processed_root = Path(processed_dir)
    processed_root.mkdir(parents=True, exist_ok=True)
    processed_path = processed_root / "house_prices_model_data.csv"
    processed_df = df.copy()
    processed_df.to_csv(processed_path, index=False)
    predictions_path = output_root / "reports" / "house_price_predictions.csv"
    pd.DataFrame(
        {
            "actual_price": np.asarray(y_test),
            "predicted_price": np.asarray(predictions),
        },
        index=y_test.index,
    ).rename_axis("row_index").to_csv(predictions_path)
    model_path = MODELS_DIR / "house_price_linear_regression.joblib"
    joblib.dump(model, model_path)

    plot_path = output_root / "plots" / "house_price_predictions.png"
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.8)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
    plt.title("Actual vs Predicted House Prices")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    report_path = output_root / "reports" / "house_price_report.json"
    report = {
        "data_file": str(csv_path),
        "rows": int(len(df)),
        "metrics": metrics,
        "output_files": {
            "processed_csv": str(processed_path),
            "predictions_csv": str(predictions_path),
            "plot": str(plot_path),
            "report": str(report_path),
            "model": str(model_path),
        },
    }
    _save_json(report_path, report)
    return report


def run_titanic_workflow(
    dataset_path: str | Path | None = None,
    output_dir: str | Path = OUTPUT_DIR,
    processed_dir: str | Path = PROCESSED_DIR,
) -> Dict[str, Any]:
    ensure_directories()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "plots").mkdir(parents=True, exist_ok=True)
    (output_root / "reports").mkdir(parents=True, exist_ok=True)

    dataset = load_titanic_dataset(dataset_path)
    X, y = prepare_titanic_data(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = Pipeline([
        ("preprocessor", _build_titanic_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = evaluate_classification(np.asarray(y_test), np.asarray(predictions))
    cm = confusion_matrix(y_test, predictions)

    processed_root = Path(processed_dir)
    processed_root.mkdir(parents=True, exist_ok=True)
    cleaned_path = processed_root / "titanic_cleaned.csv"
    cleaned_df = dataset.copy()
    for column in ["age", "fare", "pclass", "sibsp", "parch"]:
        cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
    for column in ["sex", "embarked"]:
        cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode().iloc[0])
    cleaned_df.to_csv(cleaned_path, index=False)
    model_path = MODELS_DIR / "titanic_logistic_regression.joblib"
    joblib.dump(model, model_path)

    plot_path = output_root / "plots" / "titanic_age_distribution.png"
    plt.figure(figsize=(8, 6))
    sns.histplot(data=cleaned_df, x="age", hue="survived", kde=True, multiple="stack")
    plt.title("Titanic age distribution by survival status")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    report_path = output_root / "reports" / "titanic_model_report.json"
    report = {
        "rows": int(len(dataset)),
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "output_files": {
            "cleaned_csv": str(cleaned_path),
            "plot": str(plot_path),
            "report": str(report_path),
            "model": str(model_path),
        },
    }
    _save_json(report_path, report)
    return report


def run_all_workflows() -> Dict[str, Dict[str, Any]]:
    house_report = run_house_price_workflow()
    titanic_report = run_titanic_workflow()
    return {"house_price": house_report, "titanic": titanic_report}


def run_week2_titanic_comparison(
    dataset_path: str | Path | None = None,
    output_dir: str | Path = OUTPUT_DIR,
    model_dir: str | Path = MODELS_DIR,
) -> Dict[str, Any]:
    """Compare Titanic classifiers; tree depth is selected only by training-fold CV."""
    ensure_directories()
    output_root = Path(output_dir)
    (output_root / "plots").mkdir(parents=True, exist_ok=True)
    (output_root / "reports").mkdir(parents=True, exist_ok=True)
    model_root = Path(model_dir)
    model_root.mkdir(parents=True, exist_ok=True)

    dataset = load_titanic_dataset(dataset_path)
    X, y = prepare_titanic_data(dataset)
    row_indices = np.arange(len(dataset))
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, row_indices, test_size=0.2, random_state=42, stratify=y
    )
    min_train_class_count = int(y_train.value_counts().min())
    if min_train_class_count < 2:
        raise ValueError("Not enough examples per class in the training split for stratified validation.")
    cv_splits = min(5, min_train_class_count)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    depth_values: List[int | None] = [1, 2, 3, 4, 5, 8, None]
    depth_results: List[Dict[str, Any]] = []
    for max_depth in depth_values:
        depth_pipeline = Pipeline(
            [
                ("preprocessor", _build_titanic_preprocessor(scale_numeric=False)),
                ("classifier", DecisionTreeClassifier(max_depth=max_depth, random_state=42)),
            ]
        )
        scores = cross_validate(
            depth_pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy",
            return_train_score=True,
            n_jobs=1,
        )
        depth_results.append(
            {
                "max_depth": "None" if max_depth is None else max_depth,
                "training_accuracy_mean": float(np.mean(scores["train_score"])),
                "validation_accuracy_mean": float(np.mean(scores["test_score"])),
                "validation_accuracy_std": float(np.std(scores["test_score"])),
            }
        )

    # Validation CV—not the holdout test set—chooses the simplest best-scoring depth.
    selected_depth_result = max(
        depth_results,
        key=lambda result: (
            result["validation_accuracy_mean"],
            -(result["max_depth"] if isinstance(result["max_depth"], int) else 10**9),
        ),
    )
    selected_depth = selected_depth_result["max_depth"]
    if selected_depth == "None":
        selected_depth = None
    depth_csv_path = output_root / "reports" / "titanic_tree_depth_validation.csv"
    pd.DataFrame(depth_results).to_csv(depth_csv_path, index=False)

    knn_neighbors = min(5, len(X_train))
    model_pipelines: Dict[str, Pipeline] = {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", _build_titanic_preprocessor(scale_numeric=True)),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("preprocessor", _build_titanic_preprocessor(scale_numeric=False)),
                ("classifier", DecisionTreeClassifier(max_depth=selected_depth, random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", _build_titanic_preprocessor(scale_numeric=False)),
                ("classifier", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)),
            ]
        ),
        "K-Nearest Neighbors": Pipeline(
            [
                ("preprocessor", _build_titanic_preprocessor(scale_numeric=True)),
                ("classifier", KNeighborsClassifier(n_neighbors=knn_neighbors)),
            ]
        ),
    }

    comparison_rows: List[Dict[str, Any]] = []
    prediction_data: Dict[str, Any] = {"row_index": test_indices, "survived_actual": y_test.to_numpy()}
    confusion_matrices: Dict[str, List[List[int]]] = {}
    classification_reports: Dict[str, Any] = {}
    saved_models: Dict[str, str] = {}
    for label, pipeline in model_pipelines.items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = evaluate_classification(y_test.to_numpy(), predictions)
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        confusion_matrices[label] = matrix.tolist()
        classification_reports[label] = classification_report(
            y_test, predictions, labels=[0, 1], output_dict=True, zero_division=0
        )
        comparison_rows.append({"model": label, **metrics})
        prediction_column = label.lower().replace(" ", "_") + "_prediction"
        prediction_data[prediction_column] = predictions
        model_filename = label.lower().replace(" ", "_").replace("-", "") + ".joblib"
        model_path = model_root / f"titanic_{model_filename}"
        joblib.dump(pipeline, model_path)
        saved_models[label] = str(model_path)

    comparison_csv_path = output_root / "reports" / "titanic_model_comparison.csv"
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(comparison_csv_path, index=False)
    predictions_csv_path = output_root / "reports" / "titanic_predictions.csv"
    pd.DataFrame(prediction_data).to_csv(predictions_csv_path, index=False)

    comparison_plot_path = output_root / "plots" / "titanic_model_comparison.png"
    comparison_df.set_index("model")[["accuracy", "f1"]].plot(
        kind="bar", figsize=(10, 6), ylim=(0, 1), rot=15
    )
    plt.title("Titanic Classifier Comparison (Stratified Test Set)")
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig(comparison_plot_path)
    plt.close()

    depth_plot_path = output_root / "plots" / "titanic_tree_depth_validation.png"
    depth_df = pd.DataFrame(depth_results)
    depth_labels = depth_df["max_depth"].astype(str)
    plt.figure(figsize=(9, 6))
    plt.plot(depth_labels, depth_df["training_accuracy_mean"], marker="o", label="Training CV folds")
    plt.plot(depth_labels, depth_df["validation_accuracy_mean"], marker="o", label="Validation CV folds")
    plt.title("Decision Tree max_depth: Training vs Validation")
    plt.xlabel("max_depth (None = unrestricted)")
    plt.ylabel("Mean accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(depth_plot_path)
    plt.close()

    confusion_plot_path = output_root / "plots" / "titanic_confusion_matrices.png"
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for axis, (label, matrix) in zip(axes.flat, confusion_matrices.items()):
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
        axis.set_title(label)
        axis.set_xlabel("Predicted survival")
        axis.set_ylabel("Actual survival")
        axis.set_xticklabels(["Died", "Survived"])
        axis.set_yticklabels(["Died", "Survived"], rotation=0)
    fig.suptitle("Titanic Test-Set Confusion Matrices")
    fig.tight_layout()
    fig.savefig(confusion_plot_path)
    plt.close(fig)

    report_path = output_root / "reports" / "titanic_week2_report.json"
    report = {
        "dataset_rows": int(len(dataset)),
        "random_seed": 42,
        "test_size": 0.2,
        "validation_strategy": f"{cv_splits}-fold stratified cross-validation on training data",
        "decision_tree_max_depth_selected_by_validation": selected_depth,
        "decision_tree_depth_validation": depth_results,
        "model_metrics": comparison_rows,
        "confusion_matrices": confusion_matrices,
        "classification_reports": classification_reports,
        "output_files": {
            "model_comparison_csv": str(comparison_csv_path),
            "predictions_csv": str(predictions_csv_path),
            "tree_depth_validation_csv": str(depth_csv_path),
            "comparison_plot": str(comparison_plot_path),
            "depth_validation_plot": str(depth_plot_path),
            "confusion_matrices_plot": str(confusion_plot_path),
            "report": str(report_path),
            "models": saved_models,
        },
    }
    _save_json(report_path, report)
    return report


def terminal_menu() -> None:
    while True:
        print("\nML Project Menu")
        print("1. Run house price workflow")
        print("2. Run Titanic workflow")
        print("3. Run all workflows")
        print("4. Exit")
        print("5. Run Week 2 (house regression + Titanic model comparison)")
        print("6. Run Week 3 (Iris clustering and PCA)")
        print("7. Run Week 4 (customer segmentation + model evaluation)")
        choice = input("Select an option: ").strip()

        if choice in {"1", "house", "house-price"}:
            report = run_house_price_workflow()
            print(json.dumps(report["metrics"], indent=2))
        elif choice in {"2", "titanic"}:
            report = run_titanic_workflow()
            print(json.dumps(report["metrics"], indent=2))
        elif choice in {"3", "all"}:
            report = run_all_workflows()
            print(json.dumps({key: value["metrics"] for key, value in report.items()}, indent=2))
        elif choice in {"4", "exit", "quit"}:
            print("Goodbye.")
            break
        elif choice in {"5", "week2", "week 2"}:
            house_report = run_house_price_workflow()
            titanic_report = run_week2_titanic_comparison()
            print(
                json.dumps(
                    {
                        "house_price_metrics": house_report["metrics"],
                        "titanic_model_metrics": titanic_report["model_metrics"],
                    },
                    indent=2,
                )
            )
        elif choice in {"6", "week3", "week 3"}:
            from src.week3_clustering import run_iris_clustering_workflow

            iris_report = run_iris_clustering_workflow()
            print(
                json.dumps(
                    {
                        "dataset_shape": [
                            iris_report["sample_count"],
                            iris_report["feature_count"],
                        ],
                        "feature_names": iris_report["feature_names"],
                        "target_names": iris_report["target_names"],
                        "cluster_centers_scaled": iris_report["cluster_centers_scaled"],
                        "cluster_centers_original_units": iris_report[
                            "cluster_centers_original_units"
                        ],
                        "predicted_cluster_labels": iris_report[
                            "predicted_cluster_labels"
                        ],
                        "pca_explained_variance_ratio": iris_report[
                            "pca_explained_variance_ratio"
                        ],
                        "pca_cumulative_explained_variance": iris_report[
                            "pca_cumulative_explained_variance"
                        ],
                        "predictions_match_after_joblib_reload": iris_report[
                            "predictions_match_after_joblib_reload"
                        ],
                        "output_files": iris_report["output_files"],
                    },
                    indent=2,
                )
            )
        elif choice in {"7", "week4", "week 4"}:
            from src.week4_capstone import run_week4_workflows

            week4_reports = run_week4_workflows()
            print(
                json.dumps(
                    {
                        "customer_segmentation": {
                            "dataset_source": week4_reports["customer_segmentation"][
                                "dataset_source"
                            ],
                            "model_metrics": week4_reports["customer_segmentation"][
                                "model_metrics"
                            ],
                            "cluster_summary": week4_reports["customer_segmentation"][
                                "cluster_summary"
                            ],
                            "output_files": week4_reports["customer_segmentation"][
                                "output_files"
                            ],
                        },
                        "svm_evaluation": {
                            "cross_validation": week4_reports["model_evaluation"][
                                "cross_validation"
                            ],
                            "grid_search": week4_reports["model_evaluation"][
                                "grid_search"
                            ],
                            "held_out_metrics": week4_reports["model_evaluation"][
                                "held_out_metrics"
                            ],
                            "output_files": week4_reports["model_evaluation"][
                                "output_files"
                            ],
                        },
                    },
                    indent=2,
                )
            )
        else:
            print("Invalid option. Please choose 1, 2, 3, 4, 5, 6, or 7.")


if __name__ == "__main__":
    terminal_menu()
