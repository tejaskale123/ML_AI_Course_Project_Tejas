# ML AI Course Project - Tejas

This project contains a small Windows-friendly machine learning workspace that demonstrates two supervised learning workflows:

- House price prediction using a regression model
- Titanic survival prediction using Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors
- Week 2 model comparison, decision-tree depth validation, model artifacts, prediction tables, and plots

The project is designed to run locally from a desktop folder without changing any existing global Python environment.

## Project structure

- `src/ml_workflows.py` - reusable ML data-prep and workflow functions
- `tests/test_ml_workflows.py` - pytest coverage for preprocessing, workflow execution, and validation checks
- `docs/week2_supervised_learning.md` - Week 2 algorithms, practice answers, and execution guide
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

## Known limitations

- The Titanic workflow depends on access to the actual Titanic dataset.
- The included Titanic data is the seaborn dataset, not Kaggle's separate `train.csv`/`test.csv` split.
- The house-price workflow is intentionally based on the provided sample CSV, so it is a small demonstration model.
- Small sample datasets can produce unstable metrics; reported scores describe only the stated split of the supplied dataset and are not claims of real-world performance.
- The course PDFs `week 2(4).pdf` and `practice set(20261013-080421).pdf` were not present in the project during this implementation. Week 2 was implemented against the requirements in the task; please add the PDFs to review any course-specific differences.

## Week 2 course material

See [docs/week2_supervised_learning.md](docs/week2_supervised_learning.md) for beginner-level explanations, practice question solutions, validation design, and artifact details.

## Notes

This project is designed to be local-only and safe for a desktop workspace. No files outside the project folder are changed.
