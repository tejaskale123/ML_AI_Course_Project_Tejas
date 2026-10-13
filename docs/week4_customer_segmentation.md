# Week 4: Customer Segmentation and Model Evaluation

This Week 4 extension remains inside the same course project. It contains two distinct exercises: the unsupervised customer-segmentation capstone and supervised cross-validation/SVM practice.

## Customer Segmentation capstone

### Dataset

The preferred dataset is the Mall Customers CSV with numeric fields `Age`, `Annual Income (k$)`, and `Spending Score (1-100)`. This project currently has no local copy. Obtain `Mall_Customers.csv` from the [Kaggle Customer Segmentation Tutorial dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) and place it at:

```text
data/raw/Mall_Customers.csv
```

The project does not silently substitute generated data for the real dataset. If the CSV is missing, the workflow creates a deterministic 150-row synthetic teaching sample in memory, records `synthetic_fallback` and its non-real provenance in the JSON report, and does not save it as if it were a source dataset. Do not use its segment findings as evidence about real customers.

### Validation and cleaning

The workflow checks that the dataset is non-empty and contains the three clustering features. Feature column labels are normalized for common Mall CSV spelling variants. Numeric values are parsed; exact duplicate rows and rows missing any clustering feature are removed. Data-quality counts, input feature data types, and post-cleaning missing-value counts are written to the report. Other columns, such as customer ID and gender, are retained in assignment output but not used as model features.

### Clustering and segment labels

The three selected numeric features are standardized, then K-Means is fitted with five clusters, `n_init=10`, and `random_state=42`. The Elbow Method records inertia from K=1 through K=10 and plots the curve. The reported elbow recommendation is a geometric heuristic; it is not an automatic guarantee of the correct business segmentation.

Cluster summaries include customer counts and mean age, annual income, and spending score. Segment descriptions are derived from the **observed cluster-average spending score**, not from arbitrary cluster IDs: below 40 means `Low Spenders`, 40 through 60 means `Medium Spenders`, and above 60 means `High Spenders`, assuming the feature is actually on a 1–100 scale. If that scale check fails, the workflow uses generic cluster names instead.

The workflow saves a scaler and K-Means model, reloads them with joblib, and predicts a hypothetical median-feature customer example. This example is explicitly labelled hypothetical; it is not a new observed customer.

### Business value and limitations

Segments can help an analyst explore groups with different income/spending patterns and frame questions about offers, service, or further research. Clusters are descriptive patterns in the selected features—not verified customer types, causal explanations, or guaranteed business outcomes. Results depend on scaling, feature choice, K, sample quality, and outliers. The Mall dataset is small and may not represent a current or specific business's customer base.

## Separate supervised practice

The evaluation exercise uses scikit-learn's built-in Wisconsin Breast Cancer dataset; it is separate from customer clustering and does not use customer observations.

- `cross_val_score` evaluates an SVM classifier with five-fold shuffled `StratifiedKFold` accuracy on the training partition.
- A stratified 80/20 train/test split reserves a held-out test set.
- `GridSearchCV` evaluates a compact grid of SVM C, kernel, and gamma values. A scikit-learn `Pipeline` contains `StandardScaler` and `SVC`, so each fold fits its scaler only on that fold's training data.
- The selected estimator is evaluated on the held-out data with accuracy, precision, recall, and a confusion matrix. It is saved, reloaded, and checked for identical predictions.
- For scikit-learn's breast-cancer dataset, class `1` is `benign`; the reported precision and recall use that positive-class convention. The report and confusion-matrix axes identify class names explicitly.

## Run and test

From PowerShell in the project root:

```powershell
$conda = "$env:USERPROFILE\miniconda3\condabin\conda.bat"
& $conda run -p .\.venv python -c "from src.week4_capstone import run_customer_segmentation; print(run_customer_segmentation())"
& $conda run -p .\.venv python -c "from src.week4_capstone import run_week4_model_evaluation; print(run_week4_model_evaluation())"
& $conda run -p .\.venv python -m pytest -q
```

Or run both workflows by selecting menu option **7** in `main.py`.

## Outputs

- Customer assignments, cluster summary, Elbow inertia, quality summary, and JSON report: `outputs/reports/week4_customer_*`
- Customer Elbow and three-feature scatter plots: `outputs/plots/week4_customer_*.png`
- SVM fold scores, GridSearchCV table, test predictions, and JSON metrics: `outputs/reports/week4_svm_*`
- SVM fold-score and confusion-matrix visuals: `outputs/plots/week4_svm_*.png`
- Customer scaler and K-Means: `models/week4_customer_scaler.joblib`, `models/week4_customer_kmeans.joblib`
- Fitted best SVM pipeline: `models/week4_svm_best_model.joblib`
