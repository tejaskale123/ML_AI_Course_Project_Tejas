# Week 2: Supervised Learning

This guide accompanies the existing project workflows. All outputs are written beneath this project; model pipelines are saved in `models/`, reports and holdout predictions in `outputs/reports/`, and plots in `outputs/plots/`.

## The five algorithms

- **Linear Regression** predicts a numeric value by fitting a linear relationship between features and a target. This project predicts house price and reports MSE, RMSE, MAE, and R².
- **Logistic Regression** estimates class probabilities and is used here for the binary `survived` target. Despite its name, it is a classifier.
- **Decision Tree** learns a sequence of feature-based splits. `max_depth` limits tree complexity; an unrestricted/deep tree can overfit.
- **Random Forest** averages many randomized decision trees. This often improves stability compared with one tree.
- **K-Nearest Neighbors (KNN)** predicts from nearby training examples. Because distance is scale-sensitive, numeric input features are standardized.

## Run the Week 2 workflows

From PowerShell in the project root, after installing `requirements.txt` into `.venv`:

```powershell
$conda = "$env:USERPROFILE\miniconda3\condabin\conda.bat"
& $conda run -p .\.venv python main.py
```

Choose **5** to run the housing Linear Regression and the Titanic four-model comparison. Or run each from PowerShell:

```powershell
& $conda run -p .\.venv python -c "from src.ml_workflows import run_house_price_workflow; print(run_house_price_workflow())"
& $conda run -p .\.venv python -c "from src.ml_workflows import run_week2_titanic_comparison; print(run_week2_titanic_comparison())"
```

The Titanic CSV must be the real dataset at `data/raw/titanic.csv`. This project uses the seaborn Titanic dataset; it does not generate or substitute Titanic records.

## Practice questions and solutions

1. **Train Logistic Regression on Titanic.** `run_week2_titanic_comparison()` creates a scikit-learn pipeline that imputes numeric and categorical values, one-hot encodes `sex` and `embarked`, scales numeric features, and fits Logistic Regression. The preprocessor is fit as part of model training on the training split only.
2. **Generate a confusion matrix and accuracy.** The workflow evaluates the holdout predictions, reports accuracy/precision/recall/F1, and saves per-model confusion matrices in `outputs/reports/titanic_week2_report.json`. The visual matrices are in `outputs/plots/titanic_confusion_matrices.png`.
3. **Compare Decision Tree and KNN accuracy.** Compare the `Decision Tree` and `K-Nearest Neighbors` rows in `outputs/reports/titanic_model_comparison.csv`. All four classifiers use the same stratified train/test partition.
4. **Tune `max_depth` and record differences.** The workflow evaluates depths 1, 2, 3, 4, 5, 8, and unrestricted using stratified cross-validation within the training partition. `titanic_tree_depth_validation.csv` records mean training-fold and validation-fold accuracy. The plot visualizes both. The depth is selected from validation accuracy (ties prefer a shallower depth); the holdout test set is not used for tuning. The selected tree's holdout score is shown alongside other final model metrics.

## Housing assignments

The existing house-price workflow trains Linear Regression using a reproducible 80/20 split. Preprocessing is fit on training data. Its JSON metrics include MSE, RMSE, MAE, and R². The actual-vs-predicted chart is `outputs/plots/house_price_predictions.png`; held-out predictions are in `outputs/reports/house_price_predictions.csv`; the full fitted pipeline is `models/house_price_linear_regression.joblib`.

The included house-price CSV is a small teaching sample. Metrics are calculated from actual model execution but must not be presented as evidence of performance on real housing markets.

## Titanic artifacts

- `outputs/reports/titanic_model_comparison.csv`: holdout metrics for Logistic Regression, tuned Decision Tree, Random Forest, and KNN.
- `outputs/reports/titanic_predictions.csv`: actual holdout labels and predictions from each classifier.
- `outputs/reports/titanic_tree_depth_validation.csv`: training-fold and validation-fold scores used to select tree depth.
- `outputs/reports/titanic_week2_report.json`: split/validation details, metrics, per-model confusion matrices, classification reports, and artifact paths.
- `outputs/plots/titanic_model_comparison.png`, `outputs/plots/titanic_tree_depth_validation.png`, and `outputs/plots/titanic_confusion_matrices.png`.
- `models/titanic_*.joblib`: fitted pipelines that include their preprocessing steps.

Run the checks with `python -m pytest -q` inside the project environment.
