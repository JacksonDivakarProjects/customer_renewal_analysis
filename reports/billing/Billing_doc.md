# Churn Analysis

## 1. Project Overview

The goal of this project is to identify customers who are likely to churn before their renewal date (6 months to 45 days), using the datasets given.

- Billing
- Renewal call
- Emails
- Cc_calls

For Billing Dataset, the work is divided into two major stages:

- Billing Data Cleaning & Preparation
- Hypothesis Testing & Feature Validation for Churn Prediction

## 2. Data Sources

### Raw Datasets

- Billing Data
- Renewal Calls Data

## 3. Stage 1 – Billing Data Cleaning

Prepare a high-quality, analysis-ready billing dataset with:

- One row per customer
- Valid renewal outcomes
- No leakage or extreme noise

### 3.1 Initial Data Inspection

- Billing dataset size: 122,082 rows × 59 columns
- Customers (`Co_Ref`) repeat across multiple renewal years
- Missing values found in several fields:
  - Connection metrics
  - Discount fields
  - Last-year payment information
- Important observation: `Co_Ref + Renewal_Year` is already unique

### 3.2 Understanding Duplicates

- Same customer appears in multiple years (2023–2026)
- Business rule: Keep the most recent renewal record per customer
- `Co_Ref + Renewal_year` was considered for understanding duplicates

### 3.3 Cleaning Strategy

The following steps were applied in sequence:

- **Step 1: Date Parsing**
  - Converted all relevant date fields into standard datetime format:
    - `Prospect_Renewal_Date`
    - `Closed_Date`
    - `Proforma_Date`
    - `Registration_Date`
    - `Last_Renewal`
- **Step 2: Remove Invalid Renewal Years**
  - Dropped clearly erroneous data:
    - Year 2027
    - Year 2050
  - Retained valid renewal years: 2023–2026
- **Step 3: Filter Valid Outcomes**
  - Kept only:
    - `Won`
    - `Churned`
    - `Open`
  - “Open” records are retained temporarily.
- **Step 4: Remove Extremely Sparse Columns**
  - Dropped columns with >85% missing values
  - Example:
    - Connection quantities
    - Discount amounts
  - Reason: These add noise and no analytical value.
- **Step 5: Missing Value Treatment**
  - Numeric columns: Filled using median values
  - Categorical columns: Filled with `Unknown`
  - Special handling:
    - `Last_Total_Net_Paid` → filled with 0
    - `Last_Connections` → filled with 0
- **Step 6: Derived Indicators**
  - Created additional binary signals:
    - `has_last_renewal`
    - `is_closed`
  - These capture customer lifecycle signals.
- **Step 7: De-duplication**
  - Final business rule: One customer = one row
  - Method:
    - Sorted by latest `Prospect_Renewal_Date`
    - Kept most recent record per `Co_Ref`

### 3.4 Final Output of Cleaning

- Final billing dataset
- 47,825 customers
- 55 clean columns
- Zero duplicate customers
- Outcome distribution:
  - Won
  - Churned
  - Open
- Saved as: `billings_clean.csv`

## 4. Stage 2 – Hypothesis Testing

Validate business assumptions about what drives churn, before applying machine learning.

## 5. Model Design Framework

- **Prediction Window**: Focus on 180 to 45 days before renewal date
- **Target Variable**:
  - `churn = 1` → `Prospect_Outcome = Churned`
  - `churn = 0` → `Prospect_Outcome = Won`

## 6. Renewal Call Window Filtering

- Join billing + renewal calls using `Co_Ref`
- Calculate:
  - `days_to_renewal = renewal_date – call_date`
- Keep calls where: `45 ≤ days_to_renewal ≤ 180`

## 7. Call Feature Aggregation

Multiple calls per customer were aggregated into one row per customer.

Generated Call Features:

- `call_count_window`
- `first_call_date`
- `last_call_date`
- `days_to_renewal_min`
- `days_to_renewal_max`
- `days_to_renewal_mean`

Ensures compatibility with billing data structure.

## 8. Feature Engineering

New explanatory variables were created.

## 9. Leakage Prevention

Removed columns that would not be available at prediction time:

- `Prospect_Outcome`
- `Closed_Date`
- Payment totals
- Renewal post-event scores

Prevents data leakage.

## 10. Hypotheses Tested

- **H1:** Lower tenure increases churn
  - Confirmed: Churned customers have significantly lower tenure.
- **H2:** Lower revenue bands churn more
  - Confirmed: Band A/B customers show higher churn rates.
- **H3:** Higher renewal score reduces churn
  - Confirmed: Higher `Total_Renewal_Score` is strongly protective.
- **H4:** More calls may indicate escalation
  - Confirmed: Call volume differences are statistically significant.
- **H5:** Longer tenure in days (`tenure_days`) is associated with lower churn
  - Confirmed.
  - There is a significant difference in `tenure_days` between churned and non-churned customers.
- **H6:** Price change ratio impacts churn
  - Confirmed.
  - There is a significant difference in `price_change_ratio` between churned and non-churned customers.
- **H7:** Proximity of last call to renewal (`days_last_call_to_renewal`) impacts churn
  - Confirmed.
  - There is a significant difference in `days_last_call_to_renewal` between churned and non-churned customers.
- **H8:** Call frequency per month (`call_freq_per_month`) impacts churn
  - Confirmed.
  - There is a significant difference in `call_freq_per_month` between churned and non-churned customers.

### Additional Validated Drivers

- Tenure (in days)
- Price change shock
- Proximity of last call to renewal
- Call frequency intensity

All hypotheses were validated using t-tests and chi-square tests with `p < 0.05`.

## 11. Feature Selection for Modeling

Final features chosen based on hypothesis results:

- `Total_Renewal_Score_New`
- `Tenure_Years`
- `Band`
- `Call_count_window`
- `Tenure_days`
- `Price_change_ratio`
- `Days_last_call_to_renewal`
- `Call_freq_per_month`

Categorical fields were label-encoded.

## 12. Final Outputs

- EDA
- Tenure & Score Analysis
- Band & Tenure Group Churn Rates
- Call Behaviour in the Window

## Feature Reference

| Feature | Description |
| --- | --- |
| `tenure_days` | Customer age in days |
| `price_change_ratio` | Year-over-year price change |
| `days_last_call_to_renewal` | How close last call was to renewal |
| `call_freq_per_month` | Normalized engagement intensity |

## Output Files

| File | Purpose |
| --- | --- |
| `billings_clean.csv` | Clean billing data |
| `billings_processed.csv` | Model-ready data |
