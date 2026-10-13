from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"


def ensure_directories() -> None:
    for directory in [RAW_DIR, PROCESSED_DIR, PLOTS_DIR, REPORTS_DIR]:
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
    required_columns = {"survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Titanic dataset missing required columns: {sorted(missing_columns)}")
    working = df.copy()
    working["survived"] = working["survived"].astype(int)
    working["age"] = working["age"].fillna(working["age"].median())
    working["fare"] = working["fare"].fillna(working["fare"].median())
    working["embarked"] = working["embarked"].fillna(working["embarked"].mode().iloc[0])
    features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    encoded = pd.get_dummies(working[features], columns=["sex", "embarked"], drop_first=False)
    return encoded, working["survived"]


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
        return pd.read_csv(raw_dataset_path)
    try:
        df = sns.load_dataset("titanic")
    except Exception:
        raise FileNotFoundError(
            "Titanic dataset is unavailable. Place it at data/raw/titanic.csv or ensure seaborn can download it."
        )
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
            "plot": str(plot_path),
            "report": str(report_path),
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
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = evaluate_classification(np.asarray(y_test), np.asarray(predictions))
    cm = confusion_matrix(y_test, predictions)

    processed_root = Path(processed_dir)
    processed_root.mkdir(parents=True, exist_ok=True)
    cleaned_path = processed_root / "titanic_cleaned.csv"
    cleaned_df = dataset.copy()
    cleaned_df["age"] = cleaned_df["age"].fillna(cleaned_df["age"].median())
    cleaned_df["fare"] = cleaned_df["fare"].fillna(cleaned_df["fare"].median())
    cleaned_df["embarked"] = cleaned_df["embarked"].fillna(cleaned_df["embarked"].mode().iloc[0])
    cleaned_df.to_csv(cleaned_path, index=False)

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
        },
    }
    _save_json(report_path, report)
    return report


def run_all_workflows() -> Dict[str, Dict[str, Any]]:
    house_report = run_house_price_workflow()
    titanic_report = run_titanic_workflow()
    return {"house_price": house_report, "titanic": titanic_report}


def terminal_menu() -> None:
    while True:
        print("\nML Project Menu")
        print("1. Run house price workflow")
        print("2. Run Titanic workflow")
        print("3. Run all workflows")
        print("4. Exit")
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
        else:
            print("Invalid option. Please choose 1, 2, 3, or 4.")


if __name__ == "__main__":
    terminal_menu()
