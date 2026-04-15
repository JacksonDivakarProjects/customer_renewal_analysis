# Customer Renewal & Churn Analysis

## Professional Technical Documentation

---

## 1. Executive Summary

This document provides a **complete, end-to-end professional documentation** of the **Customer Call Center (CC) Data Cleaning**, **Dataset Integration**, and **Churn Hypothesis Testing Framework** used in the customer renewal analysis project. The objective is to transform raw operational data into analytically reliable datasets, validate key churn drivers using statistical methods, and derive actionable features for downstream churn modeling.

The workflow follows industry-standard analytics practices:

- Robust data quality assessment and remediation
- Statistically sound hypothesis testing
- Effect size interpretation beyond p-values
- Feature engineering grounded in business insight

---

## 2. Data Sources Overview

### 2.1 Call Center Calls Dataset (`cc_calls.csv`)

Contains detailed records of inbound and outbound customer service interactions, including:

- Care package discussions
- Contractor sentiment and scores
- Pricing conversations
- Complaints and hardship indicators

**Initial shape:** 32,882 rows × 33 columns

### 2.2 Renewal Dataset (`renewal_calls_data.csv`)

Contains renewal-level outcomes and timelines:

- Renewal date
- Prospect outcome (Won / Churned)
- Discount activity
- Complaint indicators

---

## 3. Data Cleaning & Preparation

### 3.1 Initial Quality Assessment

- Comprehensive null-value profiling was conducted per column
- High-null columns (e.g., sentiment issue scores) were retained due to analytical relevance
- Binary fields showed inconsistent encodings ("Yes/No", mixed text, blanks)

### 3.2 General Cleaning Actions

- Missing categorical values filled as **"Not Mentioned"** where contextually valid
- Duplicate records removed (93 duplicates)
- Contact IDs cast to nullable integer type
- Call dates standardized to datetime format

**Post-cleaning shape:** 32,789 rows × 33 columns

---

## 4. Column-Specific Cleaning Logic

### 4.1 Care Package Normalization (`cc_care_package`)

**Issues Identified:**

- Multiple values in single cells
- Non-standard phrases such as "Not discussed – customer unsure"

**Resolution Strategy:**

- Multiple values → `Multiple Packages`
- All variants normalized to lowercase, trimmed, standardized

### 4.2 Binary Feature Standardization

Columns such as:

- `cc_pricing_mentioned`
- `cc_refund_discussed`
- `cc_external_consultant`

Were normalized to:

- `Yes`, `No`, `Not Applicable`

Invalid or mixed entries were coerced to NaN to prevent analytical distortion.

### 4.3 Sentiment Fields

#### Sentiment Categories

- Validated against: `Satisfied`, `Neutral`, `Dissatisfied`
- Invalid labels removed

#### Sentiment Scores

- Converted to numeric
- Missing scores filled using **median imputation** (chosen to preserve distribution and reduce bias)

---

## 5. Data Optimization

- All categorical columns converted to `category` dtype
- Memory reduced from ~8.5 MB to ~3.7 MB
- Dataset optimized for statistical testing and modeling

---

## 6. Dataset Integration

### 6.1 Merge Strategy

- **Join type:** Left Join
- **Key:** `Co_Ref`
- **Base table:** Renewal dataset
- **Supplementary table:** Cleaned CC calls dataset

**Rationale:** Retain full renewal population while appending call behavior where available.

---

## 7. Churn Analysis Framework

### 7.1 Modeling Window Definition

- Only calls occurring **45–180 days before renewal** were included
- Metric: `days_since_last_call`

**Final analytic population:** 50,333 renewal records

**Overall churn rate:** 15.12%

---

## 8. Hypothesis Testing Summary

| Hypothesis | Feature | Statistical Test | Result |
|-----------|---------|------------------|--------|
| H1 | Desire to Cancel | Chi-Square | Significant |
| H2 | Discount Offered | Chi-Square + Fisher | Significant (Reactive Discounting) |
| H3 | Contractor Sentiment Score | Mann–Whitney U | Significant |
| H4 | Pricing Mentioned | Chi-Square | Significant |
| H5 | Agent Renewal Initiation | Chi-Square | Not Significant |
| H6 | Financial Hardship | Chi-Square | Significant |
| H7 | Serious Complaint | Fisher Exact | Significant |
| H8 | Days Since Last Call | Mann–Whitney U | Significant |

---

## 9. Key Findings & Interpretations

### 9.1 Highest Churn Risk Drivers

- Explicit cancel intent (churn rate ≈ 69%)
- Financial hardship mentions
- Serious complaints
- Low sentiment scores

### 9.2 Counter-Intuitive Insights

- **Discounted customers churn more**, indicating reactive retention behavior
- Agent-initiated renewal did not show significant independent impact

---

## 10. Feature Engineering for Modeling

Derived predictors include:

### 10.1 Behavioral Flags

- Cancel intent flag & score
- Pricing risk flag
- Financial hardship flag
- Complaint severity score

### 10.2 Sentiment Features

- Sentiment bands
- Sentiment delta (start → end)
- Improvement / deterioration indicators

### 10.3 Call Timing Metrics

- Days since last call
- Call volume bands

---

## 11. Composite Churn Risk Score

A weighted, interpretable risk score was created:

**Inputs:**

- Cancel intent severity
- Sentiment score
- Discounts, hardship, complaints
- Call frequency

**Output Bands:**

- Low
- Medium
- High
- Critical

**Validation:**

- Critical band churn rate ≈ 64%
- Clear monotonic risk separation observed

---

## 12. Outputs & Deliverables

- `cc_calls_cleaned.csv` – fully cleaned CC dataset
- `cc_calls_hypothesis.csv` – modeling-ready feature set
- Visual hypothesis validation plots
- Churn risk scoring framework

---

## 13. Conclusion

This analysis establishes a **statistically validated, business-aligned churn intelligence framework**. It demonstrates that customer churn is primarily driven by behavioral signals (cancel intent, sentiment, hardship) rather than surface-level interventions like discounts.

The resulting features and risk score are production-ready and suitable for:

- Predictive churn modeling
- Targeted retention strategies
- Proactive customer intervention pipelines

