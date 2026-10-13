# Week 4 Capstone Report: Customer Segmentation Using Clustering

## Project title and objectives

**Customer Segmentation Using Clustering** explores whether customers can be grouped by age, annual income, and spending score. The project applies feature scaling, the Elbow Method, K-Means, cluster summaries, and joblib persistence. It separately practices model evaluation and SVM optimization.

## Dataset source and description

The preferred capstone data is the Mall Customers Dataset, commonly distributed as `Mall_Customers.csv` with age, annual income, and spending-score columns. A local copy is not currently included. The source and setup instruction are documented in [week4_customer_segmentation.md](./week4_customer_segmentation.md); download it from Kaggle and place it at `data/raw/Mall_Customers.csv`.

Until that file is supplied, the workflow uses a reproducible, deterministic synthetic teaching fallback. The report records this source explicitly. Synthetic fallback results are software demonstrations only and are not claims about real mall customers.

The separate evaluation exercise uses scikit-learn's built-in Wisconsin Breast Cancer classification dataset.

## Cleaning and preprocessing

The customer workflow validates required numeric columns, normalizes common column name variants, parses numeric values, removes exact duplicate rows, and drops rows missing selected clustering features. It records data-quality counts and feature types. `CustomerID` and `Gender`, if present, are retained for context but excluded from clustering.

`StandardScaler` is fit on the selected feature matrix before K-Means. The built-in sample is an unsupervised demonstration; the synthetic fallback is not split into train and test. The scaler and K-Means model are saved separately, reloaded, and used to predict a clearly marked hypothetical median-feature profile.

## Algorithms and rationale

- **K-Means** is an introductory centroid-based clustering method that groups observations by feature-space distance.
- **StandardScaler** gives age, income, and spending comparable numeric scale before Euclidean-distance clustering.
- **Elbow Method** compares inertia across K values. Its bend is a heuristic to discuss, not a guaranteed optimum.
- The **silhouette score** provides a descriptive compactness/separation metric for the fitted clusters.
- **SVM** and `GridSearchCV` appear in the distinct supervised practice task, not in customer segmentation.

## Evaluation methodology and results

The executed run for this report used the reproducible synthetic fallback because no Mall Customers CSV was present. It processed 150 rows and retained all 150 (no exact duplicates or missing clustering features were found). With the fixed seed, five-cluster K-Means produced inertia **63.9218** and silhouette score **0.5618**. A geometric elbow heuristic suggested K=4 while the configured capstone model used K=5, illustrating that the curve is a prompt for inspection rather than an automatic choice.

In that synthetic run, the fitted cluster counts and mean spending scores were:

| Cluster ID | Customers | Mean spending score | Rule-based label |
|---:|---:|---:|---|
| 0 | 30 | 75.37 | High Spenders |
| 1 | 30 | 23.00 | Low Spenders |
| 2 | 33 | 34.79 | Low Spenders |
| 3 | 31 | 73.71 | High Spenders |
| 4 | 26 | 49.92 | Medium Spenders |

These are generated-sample statistics, not real customer findings. The full report records the run's provenance and all artifacts in:

```text
outputs/reports/week4_customer_segmentation_report.json
outputs/reports/week4_customer_cluster_summary.csv
outputs/reports/week4_customer_data_quality.json
```

For this separate supervised exercise, the built-in 569-row Wisconsin Breast Cancer dataset was split stratified 80/20. Five-fold stratified CV on the training partition had accuracies **0.9451, 0.9780, 0.9670, 0.9670, and 0.9890** (mean **0.9692**). Grid search selected a scaled linear SVM with `C=0.1`; its best CV accuracy was **0.9758**. Held-out accuracy was **0.9825**, precision **0.9861**, and recall **0.9861** (precision and recall treat `benign`, target 1, as the positive class). The held-out confusion matrix, ordered by class names `[malignant, benign]`, was `[[41, 1], [1, 71]]`. These are evaluation results for this dataset and fixed split, not a clinical claim.

The observed customer results must be interpreted according to the report's `dataset_source_type`. When it says `synthetic_fallback`, the results describe only the generated teaching sample and must not be represented as Mall Customers findings. The elbow heuristic and business segment labels should also be reviewed before use.

The separate SVM report at `outputs/reports/week4_svm_evaluation_report.json` records individual stratified cross-validation scores, their mean, chosen grid parameters and CV score, held-out accuracy/precision/recall/confusion matrix, and prediction-reload verification. Scaling is kept inside the estimator pipeline so each CV fold fits its own scaler using that fold's training data.

## Graphs and interpretation

- `outputs/plots/week4_customer_elbow_method.png` compares inertia over K.
- `outputs/plots/week4_customer_segments.png` shows clusters across age/income, income/spending, and age/spending.
- `outputs/plots/week4_svm_confusion_matrix.png` compares held-out SVM predictions with actual classes.
- `outputs/plots/week4_svm_cross_validation_scores.png` shows accuracy per fold.

Cluster IDs are arbitrary. The project derives Low/Medium/High Spenders only from the mean spending score with its documented 1–100 thresholds; it does not infer the segment from a numeric cluster ID. A grouping is not proof of a stable customer persona or a successful marketing strategy.

## How to run and load the models

Run both Week 4 exercises using menu option **7**, or use:

```powershell
$conda = "$env:USERPROFILE\miniconda3\condabin\conda.bat"
& $conda run -p .\.venv python -c "from src.week4_capstone import run_week4_workflows; print(run_week4_workflows())"
```

The segmentation API `predict_customer_clusters()` loads the scaler and clusterer from `models/` and accepts a DataFrame with `Age`, `Annual Income (k$)`, and `Spending Score (1-100)`. The SVM artifact is a fitted pipeline at `models/week4_svm_best_model.joblib`; use `joblib.load()` and pass rows with the same 30 feature columns as the built-in breast-cancer dataset.

## Limitations, reproducibility, and future improvements

All random processes use seed 42. Results are reproducible in a compatible scikit-learn environment, although library-version changes can affect exact estimator behavior. The synthetic fallback is not real data. The actual Mall dataset is small and may not represent another population; K-Means is sensitive to selected features, outliers, scaling, and K. Future work could repeat segmentation on a real business dataset, validate stability across samples, involve domain experts in feature/label interpretation, and evaluate segments against outcomes before taking action.
