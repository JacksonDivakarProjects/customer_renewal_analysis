# Revenue Data Analysis Documentation

## Overview

This project performs end-to-end data preparation and statistical analysis on customer renewal call data to understand and predict membership churn. The work is split into two notebooks: data cleaning and feature engineering (`01`), followed by hypothesis testing and preprocessing for machine learning (`02`).

The target variable is `Prospect_Outcome` — whether a customer renews or churns — and the Analytical Base Table contains approximately 35,839 unique customer records.

---

## Data Sources

Four raw CSV files are loaded from a path defined in a `.env` file (`RAW_DATA_FOLDER`):

- `billings` — billing and contract history per customer
- `cc_calls` — customer care call records
- `emails` — email interaction records
- `renewal_calls` — proactive renewal call records (primary dataset)

Files are sorted before reading to ensure idempotency. A custom utility module (`src/cleaning_utils`) provides helper functions `parse_mixed_dates` and `clean_yes_no_column`.

---

## Hypothesis Testing Results

This section is placed first because the statistical findings are the primary analytical output and directly inform feature selection for downstream machine learning.

### Methodology: Effect Sizes Over P-Values

With over 35,000 records, even a trivial real-world difference will produce a statistically significant p-value. The feature selection strategy therefore prioritizes effect sizes and confidence intervals:

- **Cramers V** for categorical variables (values above 0.3 are considered large)
- **Cohens d** for continuous variables (values above 0.8 are considered large)
- **Rank-biserial correlation** for non-parametric tests

### Hypothesis 1: Desire to Cancel Predicts Churn (STRONG)

| Group | Won | Churned | Churn Rate |
|---|---|---|---|
| Cancel | 760 | 2,239 | 74.7% |
| Renew | 11,441 | 450 | 3.8% |

- Chi-square = 8130.36, p < 0.0001
- Cramers V = 0.7388 (large effect)

Customers who explicitly expressed a desire to cancel churned at nearly 20 times the rate of those who expressed intent to renew. This is the strongest individual predictor found in the dataset.

### Hypothesis 4: Customer Tenure Predicts Renewal (STRONG)

| Test | Statistic | p-value | Cohens d |
|---|---|---|---|
| Welch t-test | 77.17 | ~0 | 0.9904 |

Renewed customers have significantly more past contracts than churned customers, with an effect size near 1.0 — the largest practical effect in the dataset. Long-term customers are far less likely to churn. Retention efforts should be concentrated on newer customers.

### Hypothesis 8: Call Volume Predicts Churn (MODERATE)

- Mann-Whitney U test, p < 0.0001
- Rank-biserial r = -0.2673 (small to medium effect)

Churned customers had a meaningfully different distribution of calls in the 135-day window. The distribution is right-skewed and was log-transformed for modelling.

### Hypothesis 9: Days Since Last Call Predicts Churn (MODERATE)

- Mann-Whitney U test, p < 0.0001
- Rank-biserial r = -0.2954 (small to medium effect)

Recency of the last call is a meaningful predictor. This feature was engineered by computing the difference between the most recent call date and the dataset-wide maximum date.

### Hypothesis 3: Discount Offered Reduces Churn (WEAK-MODERATE)

- Chi-square = 379.66, p < 0.0001
- Cramers V = 0.1363 (small to medium effect)

Among customers offered a discount, the churn rate was approximately 15.6%. A statistically significant association exists, but the effect size is modest, meaning discounts alone are not a reliable retention lever.

### Hypothesis 5: Customer Asked for Justification (NEGLIGIBLE)

- Chi-square = 19.67, p < 0.001
- Cramers V = 0.0303 (negligible effect, explains ~0.09% of variance)

Statistically significant only due to sample size. Not a meaningful standalone predictor.

### Hypothesis 6: Explicit Switching Intent (NEGLIGIBLE)

- Chi-square = 32.18, p < 0.0001
- Cramers V = 0.0397 (negligible effect)

Similar to Hypothesis 5 — significant by p-value but practically unimportant.

### Hypothesis 7: Serious Complaint Predicts Churn (NEGLIGIBLE)

- Chi-square = 32.79, p < 0.0001
- Cramers V = 0.0412 (negligible effect)

Statistically significant but explains only 0.17% of variance.

### Hypothesis 2: Competitor Mentions Predict Churn (REJECTED)

- Mann-Whitney U, rank-biserial r = -0.0832 (very small, negative direction)

The hypothesis is rejected. Contrary to expectation, renewed customers showed slightly higher competitor mention counts than churned customers. Most customers had zero competitor mentions; the feature adds minimal signal.

---

## Data Cleaning Pipeline (Notebook 01)

### Feature Documentation (renewal_calls)

| Column | Type | Missing % | Cleaning Action |
|---|---|---|---|
| `Call_ID` | float64 | 0% | Used for counting only; dropped before aggregation |
| `Call_Direction` | object | 0% | Standardized `OUT_BOUND` / `IN_BOUND` to `Outbound` / `Inbound` |
| `Co_Ref` | object | 3.96% | Rows with null Co_Ref dropped (primary join key) |
| `Call_Date` | object | 0% | Parsed to datetime using `parse_mixed_dates` |
| `Churn_Category` | object | 95.76% | Filled with `Unclassified` |
| `Complaint_Category` | object | 89.81% | Filled with `Unclassified` |
| `Customer_Reaction_Category` | object | 87.62% | Filled with `Not Mentioned` |
| `Agent_Renewal_Pitch_Category` | object | 71.19% | Filled with `Not Mentioned` |
| `Customer_Renewal_Response_Category` | object | 70.88% | Filled with `Not Mentioned` |
| `Agent_Response_Category` | object | 71.06% | Filled with `Not Mentioned` |
| `Membership_Renewal_Decision` | object | 53.59% | Cleaned to `Yes` / `No` / `N/A` |
| `Competitor_Value_Comparison` | object | mixed | Regex extraction (see below) |
| `Competitor_Benefits_Mentioned` | object | mixed | Keyword extraction + MultiLabelBinarizer |
| `Price_Range_Mentioned` | object | mixed | Parsed to `min_price_mentioned`, `max_price_mentioned` |
| `Desire_To_Cancel` | object | mixed | Mapped to standard categories |

### Step-by-Step Cleaning

**Structural fixes**

An empty trailing column (`Unnamed: 20`) was dropped. Rows with a null `Co_Ref` were removed because this is the primary customer key — a null value would corrupt any downstream join or aggregation.

**Standardizing Call Direction**

`OUT_BOUND` and `IN_BOUND` were normalized to `Outbound` and `Inbound`. Rows with values outside these two categories were filtered out.

**Filling Missing Categorical Values**

A reusable `fill_missing_categories` function fills nulls with a semantically appropriate label. Columns with no recorded activity are filled with `Not Mentioned`, while columns representing a classification of churn or complaint type are filled with `Unclassified`.

**Binary (Yes/No) Columns**

Seven binary columns were cleaned using `clean_yes_no_column` from the utility module, normalizing inconsistent representations to `Yes`, `No`, or `N/A`.

**Competitor Value Comparison**

This column was structurally corrupted — it contained instructions mixed with answers, free text, multiple values in one string, and invalid formats. A priority-ordered extraction function resolves entries to one of five clean labels: `Better Value`, `Similar Value`, `Not Discussed`, `No`, `Price Mentioned`, or `Unknown`.

**Competitor Benefits Mentioned**

A keyword-matching function maps free text to five benefit categories: `price`, `service`, `offering`, `process`, and `recognition`. The resulting multi-label lists are then binarized using `MultiLabelBinarizer`, producing one binary column per category. The original text column is dropped.

**Price Range Mentioned**

A regex parser extracts numeric values and returns a `(min, max)` tuple. These become two new columns. The midpoint is then computed as a derived feature.

**Desire to Cancel**

Free text is mapped to five standard categories: `Cancel`, `Renew`, `Undecided`, `Not Discussed`, and `Mixed`. `Mixed` applies when both `cancel` and `renew` appear in the same entry.

---

## Feature Engineering (Notebook 01)

### Mid-Price and Log Transformation

The raw price midpoint (`mid_price`) had extreme skew with a maximum value of 143,880 against a median of 589. To address this, the price was capped at 5,000 and log-transformed using `log1p`. A quantile-based bucketing scheme (`price_bucket`) was also created using the quartile boundaries as bin edges.

A missingness flag (`mid_price_log_flag`) was created: `0` if a price was ever discussed in any call for that customer, `1` if never discussed. Missing price values are intentionally retained because the absence of a price discussion is itself a meaningful customer intent signal.

### Days Since Last Call

After aggregation to customer level, `days_since_last_call` is computed as the difference between each customer's most recent call date and the dataset-wide maximum date. This recency feature captures how recently the customer was engaged before the renewal decision.

### Columns Dropped Before Aggregation

The following columns were removed to avoid redundancy or because they serve no modelling purpose: `mid_price`, `mid_price_capped`, `min_price_mentioned`, `max_price_mentioned`, `Analysed_Call`, `Call_Number`, `Desire_To_Cancel` (replaced by `desire_to_cancel_clean`), and `Price_Range_Mentioned`.

---

## Aggregation to Customer Level (Notebook 01)

Because multiple calls exist per customer, the data is aggregated to one row per `Co_Ref`. Records are sorted chronologically before aggregation so that `last` correctly captures the final known state before the 45-day cutoff.

Aggregation rules by column type:

- `Call_ID` — count (total calls in the 135-day window)
- `Call_Date` — max (most recent call date)
- Sentiment and intent columns — last (final recorded state)
- Price log — mean (average price discussed, NaN-safe)
- Missingness flag — min (0 if price was ever discussed)
- Competitor benefit categories — sum (accumulated mentions across all calls)
- Complaint and agent action columns — last

Columns excluded from aggregation and the reasons for their exclusion are documented in the notebook. Key exclusions include columns with over 89% missingness (`Churn_Category`, `Complaint_Category`, `Justification_Category`, etc.) and pure identifier columns (`Call_ID`, `Co_Ref`).

---

## Billings Data Cleaning and Join (Notebook 01)

The billings table was deduplicated using `Co_Ref` + `Renewal_Year` as a composite unique key. Before deduplication, historical features were extracted:

- `Total_Past_Contracts` — count of all historical billing records per customer, used as a tenure proxy

The `Prospect_Outcome` column had an `Open` value that was remapped to `Won` before use. After deduplication, three tables are joined on `Co_Ref`:

1. Customer-level aggregated call features
2. Billing history features (`Total_Past_Contracts`)
3. Deduplicated billings (providing the `Prospect_Outcome` target label)

The join type is inner, ensuring only customers present in all three tables are included. The final cleaned dataset is exported to `CLEAN_DATA_FOLDER/renewal_clean.csv`.

---

## Preprocessing for Machine Learning (Notebook 02)

### Target Encoding

`churn` is created as a binary integer column: `1` if `Prospect_Outcome == 'Churned'`, `0` otherwise.

### Log Transformations

`Total_Calls_In_Window` and `days_since_last_call` are right-skewed and were log-transformed using `log1p` to produce `log_calls` and `log_days`.

### Encoding Categorical Variables

- `desire_to_cancel_clean` is ordinally encoded: `Renew=0`, `Not Discussed=1`, `Undecided/Mixed=2`, `Cancel=3`
- Seven binary columns are converted to 0/1 integer flags
- `Membership_Renewal_Decision` is binarized to `renewal_decision_bin`
- `Customer_Reaction_Category` and `Agent_Response_Category` are one-hot encoded after grouping categories with less than 1% frequency into an `Other` bucket. The `drop_first=True` option is applied to avoid multicollinearity.

### Column Name Standardization

All column names are lowercased and special characters (`/`, `-`, spaces, parentheses) are replaced with underscores for compatibility with downstream libraries.

### Final Feature Set

The processed dataset is saved to `PROCESS_DATA_FOLDER/renewal_processed.csv` with the following selected columns:

- Identifiers: `co_ref`
- Log-transformed numeric: `log_calls`, `log_days`, `mid_price_log`, `mid_price_log_flag`
- Competitor mentions: `offering`, `price`, `process`, `recognition`, `service`, `total_comp_mentions`
- Binary flags: justification asked, switching intent, price discussion, discount offered, serious complaint, other complaint, discount or waiver requested, renewal decision
- Ordinal: `desire_code`
- One-hot encoded: customer reaction categories, agent response categories
- Tenure: `total_past_contracts`
- Target: `churn`

---

## Output Files

| File | Location | Description |
|---|---|---|
| `renewal_clean.csv` | `CLEAN_DATA_FOLDER` | Customer-level aggregated data with target label, before ML preprocessing |
| `renewal_processed.csv` | `PROCESS_DATA_FOLDER` | Final ML-ready dataset with encoded and transformed features |