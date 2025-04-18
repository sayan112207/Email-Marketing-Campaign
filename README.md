## Email-Marketing-Campaign

This repository contains a machine learning model that predicts email click-through rates to optimize marketing campaign effectiveness. The model analyzes email engagement patterns and identifies high-probability user segments, potentially increasing click-through rates by over 45x compared to untargeted campaigns.

## Overview

The system analyzes email engagement data to:
1. Calculate baseline open and click rates
2. Build a predictive model for email clicks
3. Simulate performance improvements with targeting
4. Identify patterns across different user segments

## Data Sources

The model uses three CSV data files:
- `email_table.csv`: Main email campaign data
- `email_opened_table.csv`: Records of opened emails
- `email_clicked_table.csv`: Records of clicked email links

## Features

The model incorporates various features for prediction:
- Email content type (generic vs. personalized)
- Email version
- Day of week
- User country
- Hour of day
- User purchase history

## Key EDA Findings and Visualizations

### 1. Email Text: Short vs Long

- **Insight**: Short emails have **slightly higher open and click rates**.
- **Interpretation**: Simpler, concise content likely engages readers more effectively.

![Short vs Long Email Engagement](https://raw.githubusercontent.com/sayan112207/Email-Marketing-Campaign/refs/heads/main/images/email_text.png)

---

### 2. Personalization Matters

- **Insight**: Emails personalized with the **user's name** perform better in terms of both open and click-through rates.
- **Interpretation**: A simple "Hey John!" can go a long way in building trust and engagement.

![Personalization Effect](https://raw.githubusercontent.com/sayan112207/Email-Marketing-Campaign/refs/heads/main/images/email_version.png)

---

### 3. Best Days to Send Emails

| Metric | Top Days | Weak Days |
|--------|----------|-----------|
| **Open Rate** | Tuesday (~12%), Wednesday (~12%) | Friday (~7.4%), Weekend (~8.7%) |
| **Click Rate** | Wednesday (~2.75%), Tuesday (~2.45%) | Friday (~1.4%), Weekend (~1.7%) |

- **Interpretation**: Mid-week (Tue–Thu) campaigns are more effective. Avoid Fridays and weekends for important emails.

- ![WeekDay effect](https://raw.githubusercontent.com/sayan112207/Email-Marketing-Campaign/refs/heads/main/images/weekday.png)

---

### 4. Optimal Sending Time (Hour of Day)

| Metric | Peak Hours | Low Hours |
|--------|------------|-----------|
| **Open Rate** | 24:00 (~16%), 10–12 AM (~12.5%–13.5%) | 20–22 (~6%) |
| **Click Rate** | 24:00 (~4.1%), 10–12 AM (~2.6–2.8%) | 20–21 (~0.8–1.2%) |

- **Interpretation**: Emails sent around **midnight** and **late morning** perform best. Avoid evening hours (8–10 PM).

- ![Hour Effect](https://raw.githubusercontent.com/sayan112207/Email-Marketing-Campaign/refs/heads/main/images/by_hour.png)

---

### 5. User Country Performance

| Country | Open Rate | Click Rate |
|---------|-----------|------------|
| UK / US | ~12% | ~2.45% |
| Spain / France | ~4% | ~0.85% |

- **Insight**: UK and US audiences are far more responsive.
- **Interpretation**: Regional segmentation is critical. Consider localized campaigns for underperforming regions.

![Open and Click Rate by Country](https://raw.githubusercontent.com/sayan112207/Email-Marketing-Campaign/refs/heads/main/images/by_country.png)

---

**Key Recommendations for Email Campaign Optimization:**

- Keep emails short and to the point.
- Use personalization (e.g., first names).
- Send emails between **10 AM and midnight**, preferably **mid-week** (Tue–Thu).
- Target responsive regions (UK/US) more aggressively; localize for others.
- Avoid evenings and weekends for crucial sends.

## Technical Implementation

### Data Preparation
- Joins engagement data from three tables
- Creates binary target variables for opened and clicked emails
- Performs one-hot encoding of categorical variables
- Addresses class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)

### Model Training
- Uses Random Forest Classifier (100 estimators)
- 80/20 train-test split with stratification
- Optimizes for click prediction (binary classification)

### Performance Metrics
- **Accuracy**: 96.19%
- **AUC**: 99.15%
- **F1 Score**: 96.23%
- **Precision**: 95.26%
- **Recall**: 97.21%

## Key Findings

### Baseline Metrics
- **Open Rate**: 10.35%
- **Click Rate**: 2.12%

### Targeting Improvement
- **Baseline CTR (random sending)**: 50.00%
- **Simulated CTR (model targeting, threshold=0.5):** 95.26%
- **Number of users targeted by model:** 19977 out of 39153

> **Simulated CTR** with targeting high-probability users: 95.26% (45x improvement)

![CTR Targetting](https://raw.githubusercontent.com/sayan112207/Email-Marketing-Campaign/refs/heads/main/images/ctr_targetting.png)

This plot evaluates the efficacy of using a machine learning model to prioritize email recipients. It compares the cumulative Click-Through Rate (CTR) when targeting users ranked by model predictions versus the baseline CTR when targeting users randomly.

- Model-based targeting outperforms random targeting significantly, especially for the **top 40–50%** of users as ranked by the model.
- The top 40% of users, when targeted, retain a **CTR close to 99%**, showcasing the model’s strong ability to identify high-interest users.
- As targeting expands beyond 50%, the CTR declines, but remains above the baseline CTR (~0.5) for most segments.
- Random targeting (represented by the red dashed line) performs consistently at the baseline level, highlighting the value of intelligent targeting.

This analysis confirms that predictive modeling can dramatically improve email marketing effectiveness. By focusing on users most likely to click, marketers can achieve much higher engagement with fewer sends.

### Segment Analysis
| Country | Email Version | Open Rate | Click Rate | Sample Size |
|---------|--------------|-----------|------------|-------------|
| ES      | Generic      | 2.89%     | 0.56%      | 4,977       |
| ES      | Personalized | 4.93%     | 1.10%      | 4,990       |
| FR      | Generic      | 2.44%     | 0.54%      | 5,033       |
| FR      | Personalized | 5.70%     | 1.07%      | 4,962       |
| UK      | Generic      | 9.53%     | 1.83%      | 9,966       |
| UK      | Personalized | 14.50%    | 3.11%      | 9,973       |
| US      | Generic      | 9.15%     | 1.73%      | 30,233      |
| US      | Personalized | 14.69%    | 3.15%      | 29,866      |

### Top Feature Importances
1. User past purchases (0.221)
2. User country - France (0.040)
3. User country - Spain (0.035)
4. Weekday - Tuesday (0.031)
5. Email version - Generic (0.030)
6. Weekday - Thursday (0.030)

## Usage

1. Install dependencies:
```
pip install pandas scikit-learn imbalanced-learn
```

2. Ensure your data files are in the correct format and location
3. Run the model:
```
python model.py
```

## Business Implications

1. **Targeting Effectiveness**: By only sending emails to users with high predicted click probabilities, the CTR can potentially increase from 2.12% to 95.26%.

2. **Personalization Impact**: Personalized emails consistently outperform generic emails across all countries:
   - 1.96x higher open rates in Spain
   - 2.33x higher open rates in France
   - 1.52x higher open rates in UK
   - 1.60x higher open rates in US

3. **Geographic Targeting**: Significant variation in engagement across countries suggests geographic targeting opportunity:
   - Highest engagement: US and UK
   - Lower engagement: Spain and France

4. **Purchase History Significance**: User purchase history is the strongest predictor of email engagement, suggesting value in segmenting based on customer lifecycle stage.
