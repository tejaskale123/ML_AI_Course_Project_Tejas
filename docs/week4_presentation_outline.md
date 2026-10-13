# Week 4 Presentation Outline: Customer Segmentation

Suggested length: 5–7 minutes. Use simple language and show the generated project plots. State clearly whether the run used the real Mall Customers file or the synthetic fallback.

## Slide 1 — Title and goal

- Customer Segmentation Using Clustering
- Goal: explore groups of customers using age, annual income, and spending score.
- This is exploratory analysis, not a promise of business results.

## Slide 2 — Data

- Preferred source: Mall Customers Dataset (`Mall_Customers.csv`).
- If that file is absent, the code reports that it used a reproducible synthetic teaching sample instead.
- Show `dataset_source_type` and the row/feature counts from `outputs/reports/week4_customer_segmentation_report.json`.

## Slide 3 — Preparing the measurements

- Check required columns, numeric values, missing rows, and duplicates.
- Scale age, income, and spending score so a large-numbered measurement does not dominate distance.
- Customer ID and gender are not clustering inputs.

## Slide 4 — Choosing and fitting clusters

- Explain that K-Means places similar rows near shared centers.
- Show `week4_customer_elbow_method.png`; describe the bend as a guide, not a guaranteed answer.
- The workflow records its chosen K, random seed, inertia, and silhouette score in the JSON report.

## Slide 5 — Reading the segments

- Show `week4_customer_segments.png` and the cluster summary CSV.
- Segment labels are assigned from mean spending score (<40 low, 40–60 medium, >60 high) only when the score is on a 1–100 scale.
- Cluster numbers themselves do not mean low/medium/high.

## Slide 6 — Model saving and prediction

- Show the saved scaler and K-Means files in `models/`.
- The workflow reloads them and predicts a hypothetical median-feature profile.
- The example is not a recorded customer.

## Slide 7 — Separate evaluation practice and limitations

- Cross-validation and SVM GridSearchCV use a separate built-in classification dataset.
- Scaling lives inside the SVM pipeline to avoid leakage between folds.
- Clustering depends on data quality, feature selection, scaling, and K. Validate patterns on real data and with domain experts before business use.

## Speaker result checklist

Before presenting numerical results, read the latest customer and SVM JSON reports. Do not quote synthetic sample metrics as Mall Customers results, and do not invent a business impact.
