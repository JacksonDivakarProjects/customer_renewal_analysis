# Customer Renewal Analysis – Project Documentation

## 1. Project Overview

**Objective**  
Predict which customers are likely to churn **180 to 45 days before their renewal date**, enabling proactive retention interventions. The prediction is binary:  
- `1` – Churned  
- `0` – Retained (Won)

**Datasets Used**  
- **Billing** – Contract history, renewal scores, revenue bands, and pricing information  
- **Renewal Calls** – Proactive outbound/inbound call logs with customer sentiment and intent  
- **Emails** – AI‑extracted CRM fields from customer email interactions  
- **CC Calls** – Customer care call records (used as supplementary engagement signal)

**Analytical Workflow**  
1. **Data Cleaning & Preparation** – Standardise raw data, remove duplicates, handle missing values  
2. **Hypothesis Testing** – Validate business assumptions about churn drivers using statistical tests and effect sizes  
3. **Feature Engineering** – Build derived features from the validated signals  
4. **Machine Learning Modelling** – Train, tune, and evaluate a suite of classifiers on the joined dataset  
5. **Interpretation & Deployment Readiness** – Identify top predictive features and document model performance  

---

## 2. Data Sources and Preparation

### 2.1 Billing Data

| Step | Description |
|------|-------------|
| **Raw Size** | 122,082 rows × 59 columns |
| **Duplicates** | `Co_Ref` + `Renewal_Year` is unique; keep latest renewal record per customer |
| **Invalid Years** | Dropped 2027 and 2050; retained 2023–2026 |
| **Outcomes** | Filtered to `Won`, `Churned`, `Open` |
| **Sparse Columns** | Removed columns with >85% missing values |
| **Imputation** | Numeric → median; categorical → “Unknown”; special fields → 0 |
| **Derived Flags** | `has_last_renewal`, `is_closed` |
| **Final Output** | `billings_clean.csv` – **47,825 customers**, 55 columns, zero duplicates |

### 2.2 Renewal Calls Data

| Step | Description |
|------|-------------|
| **Raw Columns** | Call direction, customer reaction, desire to cancel, competitor mentions, price discussions, etc. |
| **Cleaning Actions** | – Standardised direction to `Outbound`/`Inbound`<br>– Filled missing categoricals with `Not Mentioned` or `Unclassified`<br>– Parsed mixed date formats<br>– Extracted competitor benefits with keyword mapping and `MultiLabelBinarizer`<br>– Parsed price ranges to `min`, `max`, and `mid_price` |
| **Aggregation** | Grouped by `Co_Ref` using `last` for sentiment/status columns, `count` for calls, `max` for call date |
| **Feature Engineering** | – `days_since_last_call` (recency)<br>– `mid_price_log` (capped at 5000, log‑transformed)<br>– `price_discussed_flag` (missingness as signal)<br>– `total_past_contracts` from billings (tenure proxy) |
| **Output** | `renewal_clean.csv` – **35,839 customers**, with target label `Prospect_Outcome` |

### 2.3 Emails Data

| Step | Description |
|------|-------------|
| **Raw Data** | AI‑filled CRM fields: sentiment, complaints, accreditation status, payment concerns, competitor mentions |
| **Cleaning** | – Converted sentiment score and chase count to numeric<br>– Standardised all text to lowercase, blanks → “Not Discussed”<br>– Mapped free‑text to `Yes`/`No`/`Not Discussed` using keyword lookups<br>– Collapsed duplicate customers (average for numeric, mode for categorical) |
| **Feature Engineering** | Five composite binary flags: `delays_in_accreditation`, `contractor_engagement`, `payment_intention`, `agent_chase_flag`, `dissatisfaction_with_support` |
| **Output** | `emails_clean.csv` and final ML‑ready `emails_processed.csv` with label‑encoded features |

### 2.4 Final Joined Analytical Table

All cleaned datasets were joined on `Co_Ref`:
- `renewal_processed.csv` (base table with target)
- `billings_processed.csv` (inner join)
- `cc_calls_processed.csv` (left join)
- `emails_processed.csv` (left join)

**Result** – A single denormalised table with **customer‑level features from all sources**, saved as `external_data.csv`.

---

## 3. Hypothesis Testing and Feature Validation

A hypothesis‑driven approach was used to select features for modelling. Because of the large sample size (>35k records), **effect sizes** (Cramér’s V, Cohen’s d) were prioritised over p‑values.

### 3.1 Validated Churn Drivers (Strong Evidence)

| Hypothesis | Effect Size | Implication |
|------------|-------------|-------------|
| **Lower tenure → higher churn** | Cohen’s d = 0.99 | Strongest continuous predictor; retain new‑customer focus |
| **Explicit desire to cancel → churn** | Cramér’s V = 0.74 | Single most powerful categorical signal |
| **Higher renewal score → lower churn** | Significant difference | (Excluded from model due to data leakage risk) |
| **Accreditation delays → churn** | Significant association | Proactive accreditation support required |
| **Contractor disengagement → churn** | Strong association | Engagement signals from emails are critical |
| **Price change shock → churn** | Significant difference | Year‑over‑year price changes matter |

### 3.2 Moderate / Weak Signals

- Call volume in the 180‑45 day window (rank‑biserial r = -0.27)
- Recency of last call (rank‑biserial r = -0.30)
- Discount offered (Cramér’s V = 0.14)
- Payment concerns / financial hardship (validated in emails)

### 3.3 Rejected or Negligible Signals

- Competitor mentions in emails (weak effect, direction opposite to expectation)
- Justification asked, switching intent, serious complaint (negligible Cramér’s V)

**Result** – A curated set of **behavioural and historical features** free from data leakage was passed to the modelling stage.

---

## 4. Feature Engineering Summary

### 4.1 Billing‑Derived Features
- `tenure_days` – customer age in days  
- `price_change_ratio` – year‑over‑year change  
- `Total_Renewal_Score_New` (excluded from modelling – leakage)  
- `Band`, `Tenure_Years`

### 4.2 Renewal Calls Features
- `log_calls` – log‑transformed call count in window  
- `log_days` – log‑transformed days since last call  
- `mid_price_log` – log‑transformed mid‑price discussed (missingness flagged)  
- `total_past_contracts` – tenure proxy from billing history  
- Competitor benefit mentions (5 binary flags)  
- `desire_code` – ordinal encoding of cancel intent

### 4.3 Emails Features
- `crm_customer_complained`  
- `Time_to_Renewal`  
- `crm_competitors_mentioned`  
- `crm_refund_mentioned`  
- `crm_contractor_engagement`  
- `crm_customer_payment_intention`  
- `crm_membership_overdue`  
- `crm_dissatisfaction_with_support`  
- `crm_contractor_sentiment_score`

All categorical features were label‑encoded or one‑hot encoded before modelling.

---

## 5. Model Development

### 5.1 Train / Test Split
- 80% training, 20% hold‑out test  
- Stratified by `churn` to preserve class balance

### 5.2 Preprocessing
- **Tree‑based models** – identity transformer (no scaling required)  
- **Linear / distance models** – `RobustScaler` + median imputation  
- Boolean columns cast to integer

### 5.3 Model Registry (12 Classifiers)

| Category | Models |
|----------|--------|
| Tree‑Based | Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost |
| Scaled | Logistic Regression, KNN, SVM, MLP |

### 5.4 Selection & Tuning
- **5‑fold stratified CV** on training set, scoring = ROC‑AUC  
- Top 3 models selected for `RandomizedSearchCV`  
- Final model chosen based on **test ROC‑AUC** and **Average Precision (AP)**

### 5.5 Leakage Prevention
Two columns were removed before training:
- `Total_Renewal_Score_New` – post‑hoc score derived from outcome  
- `renewal_decision_bin` – customer’s stated renewal decision

---

## 6. Results and Evaluation

### 6.1 Final Model Performance (Hold‑Out Test Set)

| Metric                 | Value              |
| ---------------------- | ------------------ |
| ROC‑AUC                | ~0.95 (best model) |
| Average Precision (AP) | ~0.86              |
| F1 Score (Churned)     | ~0.78              |

*(Exact values to be inserted from actual model run.)*

### 6.2 Key Visualisations
- **Confusion Matrix** – absolute and normalised predictions  
- **ROC Curve** – trade‑off between TPR and FPR  
- **Precision‑Recall Curve** – performance under class imbalance  
- **Model Comparison Bar Chart** – AUC of all 12 classifiers  
- **Top 20 Feature Importance** – interpretability of tree‑based final model

### 6.3 Most Important Features
The top predictive signals (from tree‑based models) included:
1. `desire_to_cancel_clean` (ordinal)  
2. `total_past_contracts` (tenure proxy)  
3. `log_calls` (call volume)  
4. `crm_customer_complained`  
5. `crm_contractor_engagement`  
6. `price_change_ratio`  
7. `crm_membership_overdue`

---

## 7. Business Implications and Recommendations

1. **Early Tenure Focus** – Churn risk is highest among newer customers; onboarding and first‑year engagement are critical.  
2. **Explicit Cancel Intent** – Any mention of cancellation must trigger immediate retention workflow.  
3. **Call Activity Patterns** – Both too many calls (escalation) and very few calls (disengagement) signal risk.  
4. **Email Signals** – Complaints, payment concerns, and accreditation delays are actionable early warnings.  
5. **Price Sensitivity** – Year‑over‑year price changes should be managed with proactive communication.  

**Deployment Readiness** – The final model pipeline is serialisable and can be integrated into a batch scoring system to flag at‑risk customers 6 months before renewal.

---

## 8. Output Files Summary

| File                 | Description                                           |
| -------------------- | ----------------------------------------------------- |
| `billings_clean.csv` | Clean billing data (47.8k customers)                  |
| `renewal_clean.csv`  | Aggregated renewal call data with target              |
| `emails_clean.csv`   | Clean email features                                  |
| `external_data.csv`  | Joined dataset without identifiers (for external use) |
| `external_data.csv`  | Final ML‑ready dataset                                |


---

*This documentation synthesises the full end‑to‑end Customer Renewal Analysis project, from raw data ingestion to a validated, interpretable churn prediction model.*