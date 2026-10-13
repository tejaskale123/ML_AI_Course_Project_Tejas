# ML AI Course Project - Tejas

This project contains a small Windows-friendly machine learning workspace that demonstrates two supervised learning workflows:

- House price prediction using a regression model
- Titanic survival prediction using Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors
- Week 2 model comparison, decision-tree depth validation, model artifacts, prediction tables, and plots
- Week 3 Iris clustering with K-Means, the Elbow Method, feature scaling, PCA, and saved/reloaded model artifacts
- Week 4 customer segmentation capstone plus a separate cross-validation/SVM evaluation exercise

The project is designed to run locally from a desktop folder without changing any existing global Python environment.

## Project structure

- `src/ml_workflows.py` - reusable ML data-prep and workflow functions
- `tests/test_ml_workflows.py` - pytest coverage for preprocessing, workflow execution, and validation checks
- `docs/week2_supervised_learning.md` - Week 2 algorithms, practice answers, and execution guide
- `docs/week3_iris_clustering.md` - Week 3 clustering/PCA explanations and execution guide
- `src/week4_capstone.py` - Week 4 customer segmentation and supervised model-evaluation workflows
- `tests/test_week4_capstone.py` - Week 4 data, clustering, SVM, artifact, and menu tests
- `docs/week4_customer_segmentation.md` - Week 4 methods and dataset setup
- `docs/week4_project_report.md` - capstone report and factual execution results
- `docs/week4_presentation_outline.md` - short beginner presentation outline
- `data/raw/` - downloaded or provided raw datasets
- `data/processed/` - cleaned/processed output CSV files
- `outputs/plots/` - generated plot images
- `outputs/reports/` - JSON reports for model metrics
- `main.py` - interactive terminal menu

## Features

- Missing-value handling for numeric and categorical columns
- Categorical encoding for sex and embarked fields
- Scaling of numeric features before model training
- Train/test splitting with deterministic random seeds
- Regression metrics for house-price prediction
- Classification metrics for Titanic survival prediction
- Training-only preprocessing for Titanic models, including missing-value imputation, one-hot encoding, and feature scaling for distance/linear models
- Stratified holdout evaluation and stratified cross-validation for decision-tree depth selection
- Saved regression/classification estimators, holdout predictions, comparison tables, and metrics
- Reproducible Iris clustering, Elbow Method analysis, two-component PCA, and cluster/species comparison
- K-Means customer segmentation with clear synthetic-data provenance when the Mall dataset is absent
- Separate stratified cross-validation and leakage-safe SVM GridSearchCV practice workflow
- Joblib saving/loading with verified repeat predictions for Week 4 models
- CSV and plot generation saved only within this project folder

## Windows setup

From PowerShell in the project root:

```powershell
cd "C:\Users\admin\OneDrive\Documents\Desktop\ML_AI_Course_Project_Tejas"
$conda = "$env:USERPROFILE\miniconda3\condabin\conda.bat"
& $conda create -y -p .\.venv python=3.11
& $conda run -p .\.venv python -m pip install --upgrade pip
& $conda run -p .\.venv python -m pip install -r requirements.txt
```

These commands assume Miniconda is installed at `$env:USERPROFILE\miniconda3`. If it is installed elsewhere, set `$conda` to that installation's `condabin\conda.bat` path.

## Running the project

```powershell
cd "C:\Users\admin\OneDrive\Documents\Desktop\ML_AI_Course_Project_Tejas"
$conda = "$env:USERPROFILE\miniconda3\condabin\conda.bat"
& $conda run -p .\.venv python main.py
```

Menu options:

1. Run house price workflow
2. Run Titanic workflow
3. Run all workflows
4. Exit
5. Run Week 2 (house regression + Titanic model comparison)
6. Run Week 3 (Iris clustering and PCA)
7. Run Week 4 (customer segmentation + model evaluation)

## Running tests

```powershell
cd "C:\Users\admin\OneDrive\Documents\Desktop\ML_AI_Course_Project_Tejas"
$conda = "$env:USERPROFILE\miniconda3\condabin\conda.bat"
& $conda run -p .\.venv python -m pytest -q
```

## Dataset setup and limitations

- The sample house-price dataset is already included at `data/raw/house_prices_sample.csv`.
- The genuine 891-row Titanic dataset used by this project is saved at `data/raw/titanic.csv`. It is the seaborn Titanic dataset, sourced from [seaborn-data/titanic.csv](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv). If the file is missing, the workflow attempts to retrieve it via seaborn (internet required).
- If internet access is unavailable, download that CSV from the linked source in a browser and save it as `data/raw/titanic.csv` inside this project. The workflow does not substitute fabricated records.
- The Titanic model requires `survived`, `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, and `embarked` columns.

## Generated outputs

Generated output files are local, ignored by Git, and created inside the project:

- `data/processed/house_prices_model_data.csv`
- `data/processed/titanic_cleaned.csv`
- `outputs/reports/house_price_predictions.csv`
- `outputs/reports/titanic_predictions.csv`
- `outputs/reports/titanic_model_comparison.csv`
- `outputs/reports/titanic_tree_depth_validation.csv`
- `outputs/plots/house_price_predictions.png`
- `outputs/plots/titanic_age_distribution.png`
- `outputs/plots/titanic_model_comparison.png`
- `outputs/plots/titanic_tree_depth_validation.png`
- `outputs/plots/titanic_confusion_matrices.png`
- `outputs/reports/house_price_report.json`
- `outputs/reports/titanic_model_report.json`
- `outputs/reports/titanic_week2_report.json`
- `models/house_price_linear_regression.joblib` and `models/titanic_*.joblib`
- Week 3 reports and predictions under `outputs/reports/iris_*.csv` and `outputs/reports/week3_iris_clustering_report.json`
- Week 3 visualizations under `outputs/plots/week3_iris_*.png`
- Week 3 fitted objects: `models/iris_scaler.joblib`, `models/iris_kmeans.joblib`, and `models/iris_pca.joblib`
- Week 4 customer reports/CSV files: `outputs/reports/week4_customer_*.csv`, `week4_customer_*_quality.json`, and `week4_customer_segmentation_report.json`
- Week 4 SVM reports/CSV files: `outputs/reports/week4_svm_*.csv` and `week4_svm_evaluation_report.json`
- Week 4 plots: `outputs/plots/week4_customer_*.png` and `outputs/plots/week4_svm_*.png`
- Week 4 models: `models/week4_customer_scaler.joblib`, `models/week4_customer_kmeans.joblib`, and `models/week4_svm_best_model.joblib`

## Known limitations

- The Titanic workflow depends on access to the actual Titanic dataset.
- The included Titanic data is the seaborn dataset, not Kaggle's separate `train.csv`/`test.csv` split.
- The house-price workflow is intentionally based on the provided sample CSV, so it is a small demonstration model.
- Small sample datasets can produce unstable metrics; reported scores describe only the stated split of the supplied dataset and are not claims of real-world performance.
- The course PDFs `week 2(4).pdf` and `practice set(20261013-080421).pdf` were not present in the project during this implementation. Week 2 was implemented against the requirements in the task; please add the PDFs to review any course-specific differences.

## Week 2 course material

See [docs/week2_supervised_learning.md](docs/week2_supervised_learning.md) for beginner-level explanations, practice question solutions, validation design, and artifact details.

## Week 3 course material

The Iris workflow uses scikit-learn's built-in 150-row Iris dataset and does not require a manual dataset download. It standardizes the four features, fits K-Means with three clusters, computes an inertia curve, fits two-component PCA, compares clusters with the known species labels, and saves/reloads the fitted scaler, clusterer, and PCA model. Cluster IDs are arbitrary and are not assumed to equal Iris species IDs.

Run Week 3 by selecting menu option **6**, or directly from PowerShell in the project root:

```powershell
$conda = "$env:USERPROFILE\miniconda3\condabin\conda.bat"
& $conda run -p .\.venv python -c "from src.week3_clustering import run_iris_clustering_workflow; print(run_iris_clustering_workflow())"
```

See [docs/week3_iris_clustering.md](docs/week3_iris_clustering.md) for the method, output descriptions, and interpretation notes.

## Week 4 capstone and practice

Select menu option **7** to run customer segmentation and the separate cross-validation/SVM exercises. The capstone looks for `data/raw/Mall_Customers.csv`. If that real dataset is not present, it uses an explicitly identified, deterministic **synthetic teaching fallback**; no generated observations are represented as actual mall customers. To use the real data, download `Mall_Customers.csv` from [Kaggle's Customer Segmentation Tutorial dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) and place it at `data/raw/Mall_Customers.csv`.

The supervised practice uses scikit-learn's built-in Wisconsin Breast Cancer classification dataset. It demonstrates stratified cross-validation with `cross_val_score`, an SVM `Pipeline` with scaling inside cross-validation, `GridSearchCV`, held-out evaluation, and joblib model reload.

```powershell
$conda = "$env:USERPROFILE\miniconda3\condabin\conda.bat"
& $conda run -p .\.venv python -c "from src.week4_capstone import run_week4_workflows; print(run_week4_workflows())"
```

See [docs/week4_customer_segmentation.md](docs/week4_customer_segmentation.md), [docs/week4_project_report.md](docs/week4_project_report.md), and [docs/week4_presentation_outline.md](docs/week4_presentation_outline.md).

## Notes

This project is designed to be local-only and safe for a desktop workspace. No files outside the project folder are changed.
