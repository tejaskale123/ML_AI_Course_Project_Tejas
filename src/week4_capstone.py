"""Week 4 customer-segmentation capstone and model-evaluation practice."""

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
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.ml_workflows import MODELS_DIR, OUTPUT_DIR, RAW_DIR, ensure_directories

RANDOM_STATE = 42
MALL_FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
MALL_SOURCE_URL = (
    "https://www.kaggle.com/datasets/vjchoudhary7/"
    "customer-segmentation-tutorial-in-python"
)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _make_synthetic_mall_sample(random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Create a labeled synthetic teaching fallback, never presented as real data."""
    rng = np.random.default_rng(random_state)
    group_centers = [
        (24, 25, 25),
        (48, 25, 75),
        (30, 55, 50),
        (52, 60, 35),
        (40, 90, 75),
    ]
    chunks: List[pd.DataFrame] = []
    customer_id = 1
    for age, income, score in group_centers:
        count = 30
        chunk = pd.DataFrame(
            {
                "CustomerID": np.arange(customer_id, customer_id + count),
                "Gender": rng.choice(["Male", "Female"], size=count),
                "Age": np.rint(rng.normal(age, 5, size=count)).clip(18, 70).astype(int),
                "Annual Income (k$)": np.rint(
                    rng.normal(income, 7, size=count)
                ).clip(10, 150).astype(int),
                "Spending Score (1-100)": np.rint(
                    rng.normal(score, 10, size=count)
                ).clip(1, 100).astype(int),
            }
        )
        chunks.append(chunk)
        customer_id += count
    return pd.concat(chunks, ignore_index=True)


def _find_mall_dataset(dataset_path: str | Path | None = None) -> Path | None:
    if dataset_path is not None:
        candidate = Path(dataset_path)
        if not candidate.is_file():
            raise FileNotFoundError(f"Mall Customers CSV not found: {candidate}")
        return candidate
    candidates = [
        RAW_DIR / "Mall_Customers.csv",
        RAW_DIR / "mall_customers.csv",
        RAW_DIR / "Mall Customers.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_mall_customers(
    dataset_path: str | Path | None = None,
) -> Tuple[pd.DataFrame, str]:
    """Load a local Mall Customers CSV or explicitly identify synthetic fallback."""
    path = _find_mall_dataset(dataset_path)
    if path is None:
        return _make_synthetic_mall_sample(), "synthetic_fallback"
    return pd.read_csv(path), "local_mall_customers_csv"


def _normalize_mall_columns(data: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "age": "Age",
        "annual income (k$)": "Annual Income (k$)",
        "annual income": "Annual Income (k$)",
        "income": "Annual Income (k$)",
        "spending score (1-100)": "Spending Score (1-100)",
        "spending score": "Spending Score (1-100)",
    }
    rename = {}
    for column in data.columns:
        key = str(column).strip().casefold()
        if key in aliases:
            rename[column] = aliases[key]
    return data.rename(columns=rename)


def prepare_mall_customers(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Validate and clean feature rows; return clean source rows and numeric X."""
    if data.empty:
        raise ValueError("Mall Customers dataset is empty.")
    frame = _normalize_mall_columns(data.copy())
    missing_columns = [column for column in MALL_FEATURES if column not in frame]
    if missing_columns:
        raise ValueError(
            "Mall Customers dataset is missing clustering columns: "
            + ", ".join(missing_columns)
        )

    original_rows = len(frame)
    original_feature_types = {
        column: str(frame[column].dtype)
        for column in MALL_FEATURES
        if column in frame.columns
    }
    duplicate_rows = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().copy()
    for column in MALL_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    missing_feature_rows = int(frame[MALL_FEATURES].isna().any(axis=1).sum())
    frame = frame.dropna(subset=MALL_FEATURES).copy()
    if frame.empty:
        raise ValueError("No complete numeric customer feature rows remain after cleaning.")
    if len(frame) < 3:
        raise ValueError("At least three complete customer rows are required for clustering.")

    features = frame[MALL_FEATURES].astype(float).reset_index(drop=True)
    clean_rows = frame.reset_index(drop=True)
    quality = {
        "input_rows": int(original_rows),
        "duplicate_rows_removed": duplicate_rows,
        "rows_missing_clustering_features_removed": missing_feature_rows,
        "rows_after_cleaning": int(len(clean_rows)),
        "clustering_features": MALL_FEATURES,
        "input_feature_data_types": original_feature_types,
        "feature_data_types": {
            column: str(features[column].dtype) for column in MALL_FEATURES
        },
        "missing_values_after_cleaning": {
            column: int(features[column].isna().sum()) for column in MALL_FEATURES
        },
    }
    return clean_rows, features, quality


def predict_customer_clusters(
    new_customers: pd.DataFrame,
    model_dir: str | Path = MODELS_DIR,
) -> np.ndarray:
    """Load the saved scaler/clusterer and predict labels for valid customer rows."""
    missing = [column for column in MALL_FEATURES if column not in new_customers.columns]
    if missing:
        raise ValueError(f"New customer input is missing required columns: {missing}")
    features = new_customers[MALL_FEATURES].apply(pd.to_numeric, errors="coerce")
    if features.empty or features.isna().any().any():
        raise ValueError("New customer input must contain complete numeric feature rows.")
    scaler = joblib.load(Path(model_dir) / "week4_customer_scaler.joblib")
    model = joblib.load(Path(model_dir) / "week4_customer_kmeans.joblib")
    return model.predict(scaler.transform(features))


def run_customer_segmentation(
    dataset_path: str | Path | None = None,
    output_dir: str | Path = OUTPUT_DIR,
    model_dir: str | Path = MODELS_DIR,
    n_clusters: int = 5,
    max_k: int = 10,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Run the Week 4 Mall Customers K-Means capstone and save all artifacts."""
    ensure_directories()
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")
    raw_data, source_type = load_mall_customers(dataset_path)
    cleaned_rows, features, quality = prepare_mall_customers(raw_data)
    distinct_feature_rows = int(features.drop_duplicates().shape[0])
    if n_clusters >= len(features):
        raise ValueError(
            "n_clusters must be less than the number of cleaned rows so silhouette "
            "evaluation remains defined."
        )
    if n_clusters > distinct_feature_rows:
        raise ValueError("n_clusters cannot exceed the number of distinct feature rows.")
    if max_k < n_clusters:
        raise ValueError("max_k must be greater than or equal to n_clusters.")

    output_root = Path(output_dir)
    reports_dir = output_root / "reports"
    plots_dir = output_root / "plots"
    model_root = Path(model_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    cluster_ids = model.fit_predict(scaled_features)
    if np.unique(cluster_ids).size < 2:
        raise ValueError("K-Means produced fewer than two distinct clusters.")

    inertia_by_k: List[Dict[str, Any]] = []
    for k in range(1, min(max_k, len(features)) + 1):
        candidate = model if k == n_clusters else KMeans(
            n_clusters=k, n_init=10, random_state=random_state
        ).fit(scaled_features)
        inertia_by_k.append({"k": k, "inertia": float(candidate.inertia_)})

    assignments = cleaned_rows.copy()
    assignments["cluster_id"] = cluster_ids
    assignments["segment_label"] = "Unclassified"
    score_name = "Spending Score (1-100)"
    cluster_summary = (
        assignments.groupby("cluster_id")
        .agg(
            customer_count=("cluster_id", "size"),
            average_age=("Age", "mean"),
            average_annual_income_k=("Annual Income (k$)", "mean"),
            average_spending_score=(score_name, "mean"),
        )
        .reset_index()
    )

    # Only assign business labels if scores look like the documented 1-100 scale.
    score_min = float(features[score_name].min())
    score_max = float(features[score_name].max())
    labels_supported = score_min >= 1 and score_max <= 100
    if labels_supported:
        def spending_segment(mean_score: float) -> str:
            if mean_score < 40:
                return "Low Spenders"
            if mean_score <= 60:
                return "Medium Spenders"
            return "High Spenders"

        cluster_summary["segment_label"] = cluster_summary[
            "average_spending_score"
        ].map(spending_segment)
        label_lookup = cluster_summary.set_index("cluster_id")["segment_label"]
        assignments["segment_label"] = assignments["cluster_id"].map(label_lookup)
    else:
        cluster_summary["segment_label"] = "Cluster " + cluster_summary[
            "cluster_id"
        ].astype(str)

    silhouette = float(silhouette_score(scaled_features, cluster_ids))
    inertia_frame = pd.DataFrame(inertia_by_k)
    elbow_recommended_k: int | None = None
    if len(inertia_frame) >= 3:
        x = inertia_frame["k"].to_numpy(dtype=float)
        y = inertia_frame["inertia"].to_numpy(dtype=float)
        x_scaled = (x - x.min()) / max(float(x.max() - x.min()), 1.0)
        y_scaled = (y - y.min()) / max(float(y.max() - y.min()), 1.0)
        distances = np.abs(
            (y_scaled[-1] - y_scaled[0]) * x_scaled
            - (x_scaled[-1] - x_scaled[0]) * y_scaled
            + x_scaled[-1] * y_scaled[0]
            - y_scaled[-1] * x_scaled[0]
        ) / np.hypot(y_scaled[-1] - y_scaled[0], x_scaled[-1] - x_scaled[0])
        elbow_recommended_k = int(x[int(np.argmax(distances))])

    assignments_path = reports_dir / "week4_customer_cluster_assignments.csv"
    assignments.to_csv(assignments_path, index=False)
    summary_path = reports_dir / "week4_customer_cluster_summary.csv"
    cluster_summary.to_csv(summary_path, index=False)
    elbow_path = reports_dir / "week4_customer_elbow_inertia.csv"
    inertia_frame.to_csv(elbow_path, index=False)
    quality_path = reports_dir / "week4_customer_data_quality.json"
    _save_json(quality_path, quality)

    elbow_plot = plots_dir / "week4_customer_elbow_method.png"
    plt.figure(figsize=(8, 5))
    plt.plot(inertia_frame["k"], inertia_frame["inertia"], marker="o", label="Inertia")
    plt.axvline(n_clusters, color="tab:red", linestyle="--", label=f"Fitted k={n_clusters}")
    plt.title("Customer Segmentation: Elbow Method")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia (within-cluster sum of squares)")
    plt.xticks(inertia_frame["k"])
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(elbow_plot, dpi=150)
    plt.close()

    customer_plot = plots_dir / "week4_customer_segments.png"
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    plot_specs = [
        ("Age", "Annual Income (k$)", "Age vs Annual Income"),
        ("Annual Income (k$)", score_name, "Annual Income vs Spending"),
        ("Age", score_name, "Age vs Spending"),
    ]
    for axis, (x_col, y_col, title) in zip(axes, plot_specs):
        sns.scatterplot(
            data=assignments,
            x=x_col,
            y=y_col,
            hue="segment_label",
            style="cluster_id",
            palette="deep",
            s=55,
            alpha=0.8,
            ax=axis,
            legend=False,
        )
        axis.set_title(title)
        axis.set_xlabel(x_col)
        axis.set_ylabel(y_col)
    segment_names = sorted(assignments["segment_label"].unique())
    palette = sns.color_palette("deep", n_colors=len(segment_names))
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=palette[index],
            label=segment,
        )
        for index, segment in enumerate(segment_names)
    ]
    fig.legend(
        handles=legend_handles,
        title="Segment",
        loc="upper center",
        ncol=max(1, min(3, len(legend_handles))),
    )
    fig.suptitle("Customer Groups by Age, Income, and Spending")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(customer_plot, dpi=150)
    plt.close(fig)

    scaler_path = model_root / "week4_customer_scaler.joblib"
    model_path = model_root / "week4_customer_kmeans.joblib"
    joblib.dump(scaler, scaler_path)
    joblib.dump(model, model_path)
    example_customer = pd.DataFrame(
        [features.median().to_dict()], columns=MALL_FEATURES
    )
    demo_prediction = predict_customer_clusters(example_customer, model_root)
    example_payload = {
        **{key: float(value) for key, value in example_customer.iloc[0].items()},
        "predicted_cluster_id": int(demo_prediction[0]),
        "note": "Hypothetical median-feature example; not an observed customer record.",
    }

    report_path = reports_dir / "week4_customer_segmentation_report.json"
    report = {
        "project": "Week 4 Customer Segmentation",
        "dataset_source": (
            str(Path(dataset_path).resolve())
            if dataset_path is not None
            else (
                str(_find_mall_dataset().resolve())
                if source_type == "local_mall_customers_csv"
                else "reproducible synthetic teaching fallback; not real customer data"
            )
        ),
        "dataset_source_type": source_type,
        "dataset_download_instructions": (
            f"Download Mall_Customers.csv from {MALL_SOURCE_URL} and save it as "
            f"{(RAW_DIR / 'Mall_Customers.csv').as_posix()}."
        ),
        "input_shape": [int(raw_data.shape[0]), int(raw_data.shape[1])],
        "cleaned_shape": [int(cleaned_rows.shape[0]), int(cleaned_rows.shape[1])],
        "feature_columns": MALL_FEATURES,
        "feature_data_types": quality["feature_data_types"],
        "missing_values_by_column": {
            column: int(raw_data[column].isna().sum())
            for column in raw_data.columns
            if raw_data[column].isna().any()
        },
        "duplicate_rows_detected": quality["duplicate_rows_removed"],
        "data_cleaning": quality,
        "random_state": random_state,
        "n_clusters": n_clusters,
        "elbow_recommended_k_heuristic": elbow_recommended_k,
        "elbow_selection_note": (
            "The elbow recommendation is a geometric heuristic. The fitted K is "
            "explicitly recorded; compare it with the curve and business purpose."
        ),
        "model_metrics": {
            "inertia": float(model.inertia_),
            "silhouette_score": silhouette,
            "customer_count": int(len(assignments)),
            "cluster_count": int(assignments["cluster_id"].nunique()),
        },
        "segment_label_rule": (
            "For scores in the 1-100 scale: mean <40 is Low Spenders, mean 40-60 "
            "is Medium Spenders, and mean >60 is High Spenders. Labels are based "
            "on each cluster's observed average spending score, never cluster ID."
            if labels_supported
            else "Spending scores were not within the 1-100 scale; descriptive "
            "spending labels were withheld and generic cluster labels were used."
        ),
        "cluster_summary": cluster_summary.to_dict(orient="records"),
        "hypothetical_new_customer_prediction": example_payload,
        "output_files": {
            "cluster_assignments_csv": str(assignments_path),
            "cluster_summary_csv": str(summary_path),
            "elbow_inertia_csv": str(elbow_path),
            "data_quality_json": str(quality_path),
            "elbow_plot": str(elbow_plot),
            "customer_segments_plot": str(customer_plot),
            "model_report": str(report_path),
            "scaler_model": str(scaler_path),
            "kmeans_model": str(model_path),
        },
    }
    _save_json(report_path, report)
    return report


def run_week4_model_evaluation(
    output_dir: str | Path = OUTPUT_DIR,
    model_dir: str | Path = MODELS_DIR,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Run breast-cancer cross-validation and leakage-safe SVM grid search."""
    ensure_directories()
    output_root = Path(output_dir)
    reports_dir = output_root / "reports"
    plots_dir = output_root / "plots"
    model_root = Path(model_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    dataset = load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="target")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )
    minimum_class_count = int(y_train.value_counts().min())
    folds = min(5, minimum_class_count)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    cv_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, gamma="scale")),
        ]
    )
    fold_scores = cross_val_score(
        cv_pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=1
    )
    cv_scores_path = reports_dir / "week4_svm_cross_validation_scores.csv"
    pd.DataFrame(
        {
            "fold": np.arange(1, len(fold_scores) + 1),
            "accuracy": fold_scores,
        }
    ).to_csv(cv_scores_path, index=False)

    search_pipeline = Pipeline([("scaler", StandardScaler()), ("svm", SVC())])
    parameter_grid = {
        "svm__C": [0.1, 1, 10],
        "svm__kernel": ["linear", "rbf"],
        "svm__gamma": ["scale", "auto"],
    }
    grid_search = GridSearchCV(
        search_pipeline,
        parameter_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
        refit=True,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    accuracy = float(accuracy_score(y_test, predictions))
    precision = float(precision_score(y_test, predictions, zero_division=0))
    recall = float(recall_score(y_test, predictions, zero_division=0))

    grid_results_path = reports_dir / "week4_svm_grid_search_results.csv"
    pd.DataFrame(grid_search.cv_results_).to_csv(grid_results_path, index=False)
    predictions_path = reports_dir / "week4_svm_predictions.csv"
    pd.DataFrame(
        {
            "true_class_id": y_test.to_numpy(),
            "true_class_name": [dataset.target_names[value] for value in y_test],
            "predicted_class_id": predictions,
            "predicted_class_name": [
                dataset.target_names[value] for value in predictions
            ],
        },
        index=y_test.index,
    ).rename_axis("row_index").to_csv(predictions_path)

    confusion_plot = plots_dir / "week4_svm_confusion_matrix.png"
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=dataset.target_names,
        yticklabels=dataset.target_names,
    )
    plt.title("Week 4 Best SVM: Held-out Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.tight_layout()
    plt.savefig(confusion_plot, dpi=150)
    plt.close()

    cv_plot = plots_dir / "week4_svm_cross_validation_scores.png"
    plt.figure(figsize=(7, 5))
    plt.bar(np.arange(1, len(fold_scores) + 1), fold_scores, color="tab:blue")
    plt.axhline(
        float(np.mean(fold_scores)),
        color="tab:red",
        linestyle="--",
        label=f"Mean = {np.mean(fold_scores):.3f}",
    )
    plt.title("SVM Stratified Cross-Validation Accuracy by Fold")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(cv_plot, dpi=150)
    plt.close()

    model_path = model_root / "week4_svm_best_model.joblib"
    joblib.dump(best_model, model_path)
    loaded_model = joblib.load(model_path)
    loaded_predictions = loaded_model.predict(X_test)
    predictions_match = bool(np.array_equal(predictions, loaded_predictions))
    if not predictions_match:
        raise RuntimeError("Reloaded SVM predictions did not match fitted model predictions.")
    new_sample_predictions = loaded_model.predict(X_test.iloc[:3])

    report_path = reports_dir / "week4_svm_evaluation_report.json"
    report: Dict[str, Any] = {
        "dataset": "scikit-learn built-in Wisconsin Breast Cancer dataset",
        "sample_count": int(len(X)),
        "feature_count": int(X.shape[1]),
        "target_names": list(dataset.target_names),
        "random_state": random_state,
        "test_size": 0.2,
        "cross_validation": {
            "method": f"{folds}-fold StratifiedKFold on training partition",
            "fold_accuracies": [float(score) for score in fold_scores],
            "mean_accuracy": float(np.mean(fold_scores)),
            "standard_deviation": float(np.std(fold_scores)),
        },
        "grid_search": {
            "parameters": parameter_grid,
            "best_parameters": grid_search.best_params_,
            "best_cv_accuracy": float(grid_search.best_score_),
        },
        "held_out_metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "precision_recall_positive_class": str(dataset.target_names[1]),
            "confusion_matrix": matrix.tolist(),
        },
        "reload_predictions_match": predictions_match,
        "new_sample_predictions": [
            {
                "row_index": int(index),
                "predicted_class_id": int(prediction),
                "predicted_class_name": dataset.target_names[prediction],
            }
            for index, prediction in zip(X_test.index[:3], new_sample_predictions)
        ],
        "interpretation": (
            "Cross-validation estimates performance across training folds. Grid "
            "search selects parameters using training folds only; the held-out "
            "test accuracy and confusion matrix are reported once for evaluation."
        ),
        "output_files": {
            "cross_validation_scores_csv": str(cv_scores_path),
            "grid_search_results_csv": str(grid_results_path),
            "predictions_csv": str(predictions_path),
            "confusion_matrix_plot": str(confusion_plot),
            "cross_validation_plot": str(cv_plot),
            "report": str(report_path),
            "model": str(model_path),
        },
    }
    _save_json(report_path, report)
    return report


def run_week4_workflows() -> Dict[str, Dict[str, Any]]:
    return {
        "customer_segmentation": run_customer_segmentation(),
        "model_evaluation": run_week4_model_evaluation(),
    }
