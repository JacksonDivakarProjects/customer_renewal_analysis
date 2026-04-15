# Reports Folder Documentation

This directory contains comprehensive documentation for each major component of the **Customer Renewal Analysis** project. Each subfolder holds detailed reports covering data cleaning, hypothesis testing, feature engineering, and model training for a specific data source or analysis stage.

## Folder Structure

```
reports/
├── billing/
│   ├── Billing_doc.docx
│   └── Billing_doc.md
├── emails/
│   └── email_results.md
├── model_training/
│   └── model_results.md
└── revenue/
    └── revenue_results.md
```

## Subfolder Descriptions

### `billing/`
Documentation for the billing dataset analysis.

| File | Description |
|------|-------------|
| `Billing_doc.docx` | Original Word document detailing the billing data cleaning process, duplicate handling, missing value treatment, and the hypothesis tests performed on billing features. |
| `Billing_doc.md` | Markdown version of the same report for easier viewing in version control and text editors. |

**Key topics covered:**
- Data cleaning steps (date parsing, invalid year removal, deduplication)
- Final dataset: 47,825 unique customers with 55 columns
- Hypothesis testing results (tenure, revenue bands, renewal score, call behaviour)
- Feature selection for downstream modelling

---

### `emails/`
Documentation for the emails CRM dataset analysis.

| File | Description |
|------|-------------|
| `email_results.md` | Complete walkthrough of the two‑stage emails pipeline: data cleaning (`01_emails_data_cleaning.ipynb`) and hypothesis testing (`02_emails_hypothesis_testing.ipynb`). |

**Key topics covered:**
- Cleaning AI‑extracted CRM fields (sentiment, complaints, accreditation status)
- Standardising free‑text responses into `Yes` / `No` / `Not Discussed`
- Composite feature engineering (e.g., `crm_dissatisfaction_with_support`)
- Chi‑square tests and Cramér’s V effect sizes for 9 churn‑related hypotheses
- Final feature selection and label encoding

---

### `model_training/`
Documentation for the machine learning modelling phase.

| File | Description |
|------|-------------|
| `model_results.md` | Detailed report of the `01_model_training.ipynb` notebook, covering the entire modelling pipeline from data joining to final evaluation. |

**Key topics covered:**
- Joining billing, renewal calls, CC calls, and emails data on `Co_Ref`
- Data leakage prevention (removal of `Total_Renewal_Score_New` and `renewal_decision_bin`)
- Train/test split (80/20 stratified)
- Preprocessing strategies (tree‑based vs. scaled models)
- Model registry of 12 classifiers with 5‑fold cross‑validation
- Hyperparameter tuning via `RandomizedSearchCV`
- Final model evaluation (ROC‑AUC, Average Precision, F1)
- Output visualisations and feature importance plots

---

### `revenue/`
Documentation for the renewal calls (revenue) dataset analysis.

| File | Description |
|------|-------------|
| `revenue_results.md` | End‑to‑end documentation of the two notebooks: `01_renewal_calls_cleaning_feature_engineering.ipynb` and `02_renewal_calls_hypothesis_testing_ml_prep.ipynb`. |

**Key topics covered:**
- Cleaning call direction, competitor mentions, price ranges, and desire‑to‑cancel fields
- Aggregation of multiple calls per customer to a single row
- Feature engineering (`log_calls`, `days_since_last_call`, `mid_price_log`, `total_past_contracts`)
- Hypothesis testing with effect sizes (Cramér’s V, Cohen’s d) to validate churn drivers
- Preprocessing for machine learning (encoding, log transforms, one‑hot encoding)
- Final output: `renewal_processed.csv` (ML‑ready dataset)

---

## Usage Notes

- All `.md` files are **Git‑friendly** and can be viewed directly on GitHub, GitLab, or any Markdown renderer.
- The `.docx` file in `billing/` is the original report; the `.md` version is provided for convenience.
- These reports serve as the **primary documentation** for the project’s analytical decisions, cleaning logic, and modelling approach. They are intended to be read alongside the Jupyter notebooks in the `notebooks/` directory.

For a high‑level project summary, refer to the main `README.md` in the repository root.