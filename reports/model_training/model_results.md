# Model Training Documentation

## Overview

This notebook (`01_model_training.ipynb`) covers the full machine learning pipeline for predicting customer churn. It takes the preprocessed output from the earlier notebooks, joins additional data sources, trains a broad registry of classifiers, tunes the top performers, and produces a final evaluated model with visualizations and feature importance analysis.

The binary target is `churn` — `1` for churned, `0` for retained.

---

## Data Sources and Joining

Four processed CSV files are loaded from `PROCESS_DATA_FOLDER`:

| File | Join Type | Description |
|---|---|---|
| `renewal_processed.csv` | Base table | Customer-level renewal call features and target |
| `billings_processed.csv` | Inner join on `Co_Ref` | Billing and contract history |
| `cc_calls_processed.csv` | Left join on `Co_Ref` | Customer care call features |
| `processed_emails.csv` | Left join on `Co_Ref` | Email interaction features |

The inner join with billings ensures only customers present in both tables are retained. Left joins for cc_calls and emails preserve all billing-matched customers, treating missing interactions as zero-activity records.

`Co_Ref` and `Prospect_Outcome` are dropped before modelling and the combined dataset is exported to `EXTERNAL_DATA_FOLDER/external_data.csv` for external use.

---

## Data Leakage Removal

Before training, a correlation check between each feature and the target was run on the training split. Two columns were identified as data leaking and removed:

- `Total_Renewal_Score_New` — a pre-computed score derived from the outcome, not a genuine input signal
- `renewal_decision_bin` — the customer's stated renewal decision, which directly encodes the target

Removing these ensures the model learns from behavioral and historical signals only, not from post-hoc labels.

---

## Train / Test Split

- Split ratio: 80% train, 20% test
- `stratify=y` is applied to preserve the class distribution across both splits
- `random_state=42` for reproducibility

---

## Preprocessing

Two preprocessing strategies are applied depending on model type:

**Tree-based models** use a passthrough identity transformer — tree algorithms do not require scaling and handle raw numeric values natively. `HistGradientBoostingClassifier` additionally handles NaN values without imputation.

**Scaled models** (Logistic Regression, KNN, SVM, MLP) use a `ColumnTransformer` pipeline that applies:
1. `SimpleImputer(strategy="median")` to handle missing values
2. `RobustScaler()` to scale numeric columns — chosen over `StandardScaler` because it is resilient to outliers by scaling around the interquartile range

Boolean columns are cast to integer before any processing step.

---

## Model Registry

Twelve classifiers are registered across two categories. All models are wrapped in `sklearn.pipeline.Pipeline` with the appropriate preprocessor.

### Tree-Based Models (no scaling required)

| Model | Key Configuration |
|---|---|
| DecisionTree | Default; used as a simple baseline |
| RandomForest | 200 estimators, all CPU cores |
| ExtraTrees | 200 estimators, all CPU cores |
| GradientBoosting | 200 estimators |
| HistGradientBoosting | 200 iterations; handles NaN natively |
| XGBoost | 200 estimators, `tree_method=hist`, logloss eval metric |
| LightGBM | 200 estimators; silent output |
| CatBoost | 200 iterations; silent output |

### Linear and Distance-Based Models (scaling required)

| Model | Key Configuration |
|---|---|
| LogisticRegression | max_iter=1000 |
| KNN | 15 neighbors |
| SVM | Probability calibration enabled |
| MLP | Hidden layers: (128, 64), max 300 iterations |

XGBoost, LightGBM, and CatBoost are loaded conditionally — if any of the three packages are not installed, the model is skipped and the remaining registry continues training without interruption.

---

## Model Selection

### 5-Fold Stratified Cross-Validation

All models are evaluated on the training set using `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Stratification ensures each fold maintains the original class ratio, which is important for an imbalanced churn dataset.

Two metrics are captured per model:

- **ROC-AUC** — primary ranking metric; measures the model's ability to separate churned from retained customers across all classification thresholds
- **F1 Score** — measures the harmonic mean of precision and recall; particularly informative when class imbalance exists

Results are ranked by mean ROC-AUC. After CV scoring, each pipeline is also fit on the full training set to prepare for hold-out evaluation.

### Hold-Out Test Evaluation

After cross-validation, every trained pipeline is evaluated on the held-out 20% test set using three metrics:

- **Test ROC-AUC** — generalization measure of discrimination
- **Test F1** — balanced precision-recall performance
- **Average Precision (AP)** — area under the Precision-Recall curve; more informative than AUC when the positive class (churned) is a minority

---

## Hyperparameter Tuning

The top 3 models by test AUC are selected for tuning using `RandomizedSearchCV`. Tuning is performed on the training set only, using the same 5-fold stratified CV scheme, with ROC-AUC as the scoring metric.

Parameter grids are defined per model:

**RandomForest / ExtraTrees**
- `n_estimators`: 200, 400, 600
- `max_depth`: None, 10, 20, 30
- `min_samples_split`: 2, 5, 10
- `min_samples_leaf`: 1, 2, 4
- `max_features`: sqrt, log2, 0.5
- `class_weight`: None, balanced

**GradientBoosting**
- `n_estimators`: 200, 400
- `learning_rate`: 0.01, 0.05, 0.1, 0.2
- `max_depth`: 3, 5, 7
- `subsample`: 0.7, 0.8, 1.0
- `max_features`: sqrt, log2, None

**HistGradientBoosting**
- `max_iter`: 100, 200, 400
- `learning_rate`: 0.01, 0.05, 0.1
- `max_depth`: None, 5, 10
- `l2_regularization`: 0.0, 0.1, 1.0

**XGBoost**
- `n_estimators`: 200, 400, 600
- `learning_rate`: 0.01, 0.05, 0.1, 0.2
- `max_depth`: 3, 5, 7, 9
- `subsample` / `colsample_bytree`: 0.6, 0.8, 1.0
- `gamma`: 0, 0.1, 0.5
- `reg_alpha` / `reg_lambda`: regularization sweep
- `scale_pos_weight`: 1, 3, 5 — handles class imbalance

**LightGBM**
- `num_leaves`: 31, 63, 127
- `min_child_samples`: 10, 20, 50
- `subsample` / `colsample_bytree`: 0.7, 0.85, 1.0
- `reg_alpha` / `reg_lambda`: regularization sweep
- `class_weight`: None, balanced

**CatBoost**
- `iterations`: 200, 400
- `depth`: 4, 6, 8, 10
- `l2_leaf_reg`: 1, 3, 5, 10
- `border_count`: 32, 64, 128

**DecisionTree**
- `max_depth`: None, 5, 10, 20
- `min_samples_split`: 2, 5, 10, 20
- `min_samples_leaf`: 1, 2, 5, 10

---

## Final Model Evaluation

The best tuned model (`final_best`) is evaluated on the held-out test set. Three outputs are produced:

### Classification Report

A per-class breakdown of precision, recall, and F1 score for both `Retained` (class 0) and `Churned` (class 1) labels.

### Visualizations

Three plots are saved as PNG files:

- `model_evaluation_plots.png` — a three-panel figure containing the confusion matrix, ROC curve, and Precision-Recall curve for the final model
- `model_comparison.png` — a horizontal bar chart comparing test ROC-AUC across all models, with the final selected model highlighted in green
- `feature_importance.png` — a horizontal bar chart of the top 20 features by importance (generated only if the final model exposes `feature_importances_`, as tree-based models do)

### Feature Importance

If the final estimator is tree-based, `feature_importances_` is extracted and sorted. The top 20 features are plotted to provide interpretability — identifying which behavioral and historical signals drive the churn prediction.

---

## Output Files

| File | Location | Description |
|---|---|---|
| `external_data.csv` | `EXTERNAL_DATA_FOLDER` | Joined and cleaned dataset without identifiers or target; for external use |
| `model_evaluation_plots.png` | Working directory | Confusion matrix, ROC curve, and Precision-Recall curve for the final model |
| `model_comparison.png` | Working directory | Bar chart comparing test ROC-AUC across all trained models |
| `feature_importance.png` | Working directory | Top 20 feature importances from the final model (tree-based models only) |