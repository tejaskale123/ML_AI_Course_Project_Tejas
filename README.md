# ML AI Course Project - Tejas

This project contains a small Windows-friendly machine learning workspace that demonstrates two supervised learning workflows:

- House price prediction using a regression model
- Titanic survival prediction using a binary classification model

The project is designed to run locally from a desktop folder without changing any existing global Python environment.

## Project structure

- `src/ml_workflows.py` - reusable ML data-prep and workflow functions
- `tests/test_ml_workflows.py` - pytest coverage for preprocessing, workflow execution, and validation checks
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

The following files are created inside the project whenever workflows are executed:

- `data/processed/house_prices_model_data.csv`
- `data/processed/titanic_cleaned.csv`
- `outputs/plots/house_price_predictions.png`
- `outputs/plots/titanic_age_distribution.png`
- `outputs/reports/house_price_report.json`
- `outputs/reports/titanic_model_report.json`

## Known limitations

- The Titanic workflow depends on access to the actual Titanic dataset.
- The included Titanic data is the seaborn dataset, not Kaggle's separate `train.csv`/`test.csv` split.
- The house-price workflow is intentionally based on the provided sample CSV, so it is a small demonstration model.

## Notes

This project is designed to be local-only and safe for a desktop workspace. No files outside the project folder are changed.
