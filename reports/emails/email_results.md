# Emails Data — Documentation

> This document explains, in simple words, what was done across the two emails notebooks:
> **data cleaning** and **hypothesis testing**.

---

## Part 1 — Data Cleaning (`01_emails_data_cleaning.ipynb`)

### Overview

We started with a raw emails CRM dataset (`emails.csv`). The data came from an LLM (AI model) that read customer emails and filled in various fields. Because the AI sometimes wrote different things for the same meaning, the data was messy and inconsistent. This notebook cleans all of that up so the data is ready to be analysed.

---

### What columns does the data have?

Each row in the dataset represents one customer interaction captured from an email. The main columns are:

| Column | What it means |
|---|---|
| `Co_Ref` | A unique ID for each customer — used to link rows together |
| `Time_to_Renewal` | How close the customer is to their renewal date |
| `crm_accreditation_completed` | Did the customer finish their accreditation? |
| `crm_timely_completion` | Did they finish it on time? |
| `crm_progress_towards_accreditation` | How far along are they? |
| `crm_delays_in_accreditation` | Were there any delays? |
| `crm_contractor_suggested_leave` | Did the contractor hint they might leave? |
| `crm_contractor_engagement` | Is the contractor engaged or not? |
| `crm_contractor_sentiment` | How is the contractor feeling? (Satisfied / Dissatisfied / Neutral) |
| `crm_contractor_sentiment_score` | A number (0–100) that measures how satisfied they are |
| `crm_dts_or_ssip_mentioned` | Was DTS/SSIP brought up in the email? |
| `crm_customer_payment_intention` | Does the customer plan to pay / renew? |
| `crm_membership_overdue` | Is the customer's membership overdue? |
| `crm_membership_level` | What membership tier are they on? |
| `crm_dissatisified_with_renewal_price` | Is the customer unhappy with the price? |
| `crm_customer_complained` | Did the customer complain? |
| `crm_refund_mentioned` | Was a refund mentioned? |
| `crm_negative_customer_experience` | Was a negative experience reported? |
| `crm_dissatisfaction_with_support` | Was the customer unhappy with support? |
| `crm_financial_hardship_mentioned` | Did the customer mention money problems? |
| `crm_competitors_mentioned` | Did the customer mention any competitors? |
| `crm_agent_chase_count` | How many times did the agent follow up? |
| `crm_auto_renewal_status` | Is the customer on auto-renewal? |
| `year` | The renewal year |

---

### What cleaning steps were done?

#### Step 1 — Fix the number columns
Two columns (`crm_contractor_sentiment_score` and `crm_agent_chase_count`) should be numbers, but the AI sometimes wrote things like "Not Discussed" instead of a number. We converted them properly to numbers. Anything that couldn't be converted was set to **0** (meaning "not recorded").

#### Step 2 — Standardise all text columns
All the text columns were made consistent:
- Converted everything to lowercase with no extra spaces
- Replaced blanks, "nan", "none", or empty values with the label **"Not Discussed"**

This means every column now has clean, uniform values.

#### Step 3 — Map messy text to Yes / No / Not Discussed
Many columns had free-text answers that needed to be grouped into three simple categories: **Yes**, **No**, or **Not Discussed**.

For example, a column might contain values like "there were frustrations", "yes", "no issues raised", "n/a", and so on. We built a lookup of keywords to figure out which category each value belongs to, using a priority order: "Not Discussed" is checked first, then "No", then "Yes".

The columns cleaned this way were:
- Delays in accreditation
- Contractor engagement
- Customer complaints
- Financial hardship
- Dissatisfaction with support
- Negative customer experience
- Refund mentioned
- Competitors mentioned

#### Step 4 — Clean contractor sentiment
The sentiment column had many different free-text values. We mapped these to four clear categories:
- **Satisfied** — contractor is happy
- **Dissatisfied** — contractor is unhappy or threatening to leave
- **Neutral** — no strong feeling either way
- **Not Discussed** — sentiment was not mentioned

#### Step 5 — Remove rows with invalid values
Three columns (`crm_dts_or_ssip_mentioned`, `crm_customer_payment_intention`, `crm_membership_overdue`) should only contain Yes / No / Not Discussed. Any rows that had something else were removed to avoid bad data getting into the analysis.

#### Step 6 — Clean membership level
The membership level column had many ways of saying the same thing (e.g. "Gold Member", "gold", "GOLD"). We standardised these into clear categories: Accredited, Assisted, Standard, Silver, Gold, Bronze, Premium, Member, In Progress, Express, Band, or Not Discussed.

#### Step 7 — Collapse duplicate customer rows
Some customers appeared more than once in the data. We combined all rows for the same customer (`Co_Ref`) into a single row:
- For **number columns**, we took the **average**
- For **text columns**, we took the **most common value**

#### Step 8 — Check for duplicates
After collapsing, we checked that no customer appeared more than once. A result of **0% duplicates** confirmed the process worked correctly.

#### Step 9 — Save the clean data
The cleaned data was saved as `emails_clean.csv` for use in the next stage.

---

## Part 2 — Hypothesis Testing (`02_emails_hypothesis_testing.ipynb`)

### What is this notebook about?

Now that we have clean data, we want to find out **which things in the emails data are actually linked to whether a customer churns (leaves) or renews**. We do this by running statistical tests on different features and measuring how strongly each one is associated with the renewal outcome.

---

### How does it work?

We merged the clean emails data with the renewal calls data using the customer ID (`Co_Ref`) as the link. This gave us one combined table with everything we know about each customer.

---

### What method was used to test the hypotheses?

We used two things together:

**1. Chi-Square Test** — This checks whether two things are related or just happening by coincidence. If the p-value is below 0.05, we say the relationship is statistically significant (i.e. not random).

**2. Cramer's V** — This measures *how strong* the relationship actually is, on a scale from 0 to 1:
- Below 0.1 → the relationship exists but is too weak to be useful
- 0.1 to 0.3 → moderate strength, worth including
- Above 0.3 → strong relationship, very useful for prediction

We used Cramer's V because with large datasets, even a tiny meaningless difference can look "significant" on a p-value alone. Cramer's V tells us if the relationship is actually meaningful in practice.

---

### Feature Engineering (before testing)

Before testing, five new "combined" features were created by merging several related columns:

| New Feature | What it captures |
|---|---|
| `crm_delays_in_accreditation` | Any sign of accreditation friction (delays, incomplete, no progress) |
| `crm_contractor_engagement` | Contractor is disengaged or hinted at leaving |
| `crm_customer_payment_intention` | Customer is unhappy with price or reluctant to pay |
| `crm_agent_chase_flag` | Agent had to follow up more than twice |
| `crm_dissatisfaction_with_support` | Any complaint about support, financial hardship, or negative experience |

These are all stored as **1 (yes, this applies)** or **0 (no, it doesn't)**.

---

### The 9 Hypotheses Tested

#### Hypothesis 1 — Overdue Membership → More Churn
Customers whose membership payment is overdue might be more likely to leave.
A grouped bar chart was used to visualise the count of Won vs. Churned outcomes across overdue categories.
**Business implication:** These customers should be contacted early with payment support conversations.
Conclusion: Reject H₀ — significant association found.
---

#### Hypothesis 2 — Customer Complained → More Churn
Customers who made a complaint during email interactions might be more likely to churn.
A normalised stacked bar chart was used to show the *proportion* of churn within each complaint group.
**Business implication:** Complaints should trigger an immediate retention workflow.
Reject H₀ — significant association found.
---

#### Hypothesis 3 — Accreditation Delays → More Churn
Contractors who had delays or problems with their accreditation might be more likely to not renew.
A heatmap was used to show the joint distribution of delays and outcomes.
**Business implication:** Contractors stuck in the accreditation process need proactive support to stay on track.
Reject H₀ — significant association found.
---

#### Hypothesis 4 — Low Contractor Engagement → More Churn
Contractors who seem disengaged or have suggested they might leave are likely to churn.
Side-by-side pie charts were used to compare the churn rate between engaged and disengaged groups.
**Business implication:** Disengaged contractors need re-engagement outreach before their renewal date.
Reject H₀ — significant association found.
---

#### Hypothesis 5 — Support Dissatisfaction → More Churn
Customers who reported any negative experience, financial hardship, or dissatisfaction with support are more likely to leave.
A diverging bar chart was used to show the difference in churn proportion between satisfied and dissatisfied groups.
**Business implication:** These customers should be escalated to senior retention agents.
Reject H₀ — significant association found.
---

#### Hypothesis 6 — Time to Renewal → Association with Churn
Whether a customer is contacted early or late relative to their renewal date may affect the outcome.
A line chart was used to trace how the churn and win rates change across different time-to-renewal windows.
**Business implication:** Knowing the riskiest time windows lets the team prioritise when to reach out.
Reject H₀ — significant association found.
---

#### Hypothesis 7 — Competitors Mentioned → More Churn
Customers who mentioned competitors in their emails might be comparing options and considering leaving.
A grouped proportion bar chart was used to compare churn rates across competitor-mention categories.
**Business implication:** Customers who mention competitors should receive a targeted response highlighting value.
Reject H₀ — significant association found.
---

#### Hypothesis 8 — Payment Concerns → More Churn
Customers who showed reluctance around paying or who were unhappy with the renewal price are more likely to churn.
Donut charts were used to compare the outcome composition between customers with and without payment concerns.
**Business implication:** Offer payment plans or early discounts to customers flagged with payment concerns.
Reject H₀ — significant association found.
---

#### Hypothesis 9 — Agent Chase Flag → More Churn
Customers who required the agent to follow up more than twice may be disengaged and heading towards non-renewal.
A horizontal stacked bar chart was used to show the outcome proportion for customers with and without the chase flag.
**Business implication:** Route accounts requiring heavy chasing to specialist retention agents earlier.
Conclusion: Fail to reject H₀.
---

### Cramer's V Correlation Heatmap

After testing all hypotheses, a full correlation heatmap was created across all selected features using Cramer's V. This helps identify:
- Features that are **strongly linked to churn** (high V with `Prospect_Outcome`)
- Features that are **too similar to each other** (high V between two predictors — keeping both would be redundant)

---

### Feature Selection

Based on the hypothesis tests and the heatmap, the following features were selected for the final model:

| Feature | Why it was selected |
|---|---|
| `Co_Ref` | Identifier |
| `crm_customer_complained` | Hypothesis 2 — complaint is a strong churn signal |
| `Time_to_Renewal` | Hypothesis 6 — timing matters |
| `crm_competitors_mentioned` | Hypothesis 7 — competitor awareness |
| `crm_refund_mentioned` | Additional dissatisfaction signal |
| `crm_contractor_engagement` | Hypothesis 4 — disengagement predicts churn |
| `crm_customer_payment_intention` | Hypothesis 8 — financial reluctance |
| `crm_membership_overdue` | Hypothesis 1 — payment friction |
| `crm_dissatisfaction_with_support` | Hypothesis 5 — support issues |
| `crm_contractor_sentiment_score` | Numeric sentiment signal |
| `Prospect_Outcome` | Target variable (what we're predicting) |

---

### Encoding

All remaining text columns were converted to numbers using **Label Encoding** so that a machine learning model can read them. `Co_Ref` (the ID) and `Prospect_Outcome` (the target) were not encoded at this stage.

---

### Saving the Final Data

The processed, encoded dataset was saved as `emails_processed.csv` — ready to be fed into a machine learning model.

---

## Summary

| Stage | What happened |
|---|---|
| Data Cleaning | Fixed numbers, standardised text, removed bad rows, collapsed duplicates |
| Feature Engineering | Built 5 new binary features from combinations of raw columns |
| Hypothesis Testing | Tested 9 hypotheses using Chi-Square + Cramer's V |
| Feature Selection | Kept only features with meaningful association to churn |
| Encoding | Converted text columns to numbers |
| Output | Saved `emails_processed.csv` for modelling |
