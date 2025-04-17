# Email-Marketing-Campaign

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
- **Simulated CTR** with targeting high-probability users: 95.26% (45x improvement)

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

## Next Steps

1. Implement A/B testing with model-based targeting
2. Explore additional features (e.g., user demographics, device type)
3. Test different classification algorithms
4. Develop a real-time scoring system for campaign optimization
5. Create a threshold sensitivity analysis to optimize for business KPIs
