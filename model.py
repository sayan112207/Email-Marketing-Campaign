import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE

# Load data
email_table = pd.read_csv("email_table.csv")
opened = pd.read_csv("email_opened_table.csv")
clicked = pd.read_csv("link_clicked_table.csv")

# Prepare target variables
email_table['opened'] = email_table['email_id'].isin(opened['email_id']).astype(int)
email_table['clicked'] = email_table['email_id'].isin(clicked['email_id']).astype(int)

# Q1: What percentage of users opened and clicked?
open_rate = email_table['opened'].mean()
click_rate = email_table['clicked'].mean()
print(f"Open rate: {open_rate:.2%}")
print(f"Click rate: {click_rate:.2%}")

# Q2: Build a model to optimize sending (predict click)
# Encode categorical variables
X = email_table.copy()
X = pd.get_dummies(X, columns=['email_text', 'email_version', 'weekday', 'user_country', 'hour'])
y = X['clicked']
X = X.drop(columns=['email_id', 'opened', 'clicked'])

# Handle class imbalance
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

# Train a classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:,1]

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# AUC
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC: {auc:.4f}")

# F1 Score
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.4f}")

# Precision
precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.4f}")

# Recall
recall = recall_score(y_test, y_pred)
print(f"Recall: {recall:.4f}")

# Q3: Estimate model improvement in CTR
baseline_ctr = y_test.mean()
print(f"Baseline CTR (random sending): {baseline_ctr:.2%}")

# Simulate sending only to users with predicted click probability above a threshold
threshold = 0.5
selected = y_pred_proba > threshold
if selected.sum() == 0:
    print("No users selected above threshold. Lower the threshold.")
else:
    simulated_ctr = y_test[selected].mean()
    print(f"Simulated CTR (model targeting, threshold={threshold}): {simulated_ctr:.2%}")
    print(f"Number of users targeted by model: {selected.sum()} out of {len(y_test)}")

# Sort users by predicted probability
sorted_indices = np.argsort(-y_pred_proba)
sorted_y = y_test.iloc[sorted_indices].reset_index(drop=True)
cum_users = np.arange(1, len(sorted_y)+1)
cum_clicks = sorted_y.cumsum()
ctr_at_k = cum_clicks / cum_users

plt.figure(figsize=(8,5))
plt.plot(cum_users / len(sorted_y), ctr_at_k, label='Model CTR')
plt.axhline(baseline_ctr, color='red', linestyle='--', label='Baseline CTR')
plt.xlabel('Fraction of users targeted (top X% by model)')
plt.ylabel('Cumulative CTR')
plt.title('Simulated CTR Uplift by Model Targeting')
plt.legend()
plt.tight_layout()
plt.show()

# Q4: Patterns in segments (example: by country and personalization)
segment_summary = email_table.groupby(['user_country', 'email_version']).agg(
    open_rate=('opened', 'mean'),
    click_rate=('clicked', 'mean'),
    count=('email_id', 'count')
).reset_index()
print(segment_summary)

# Optional: Feature importance
importances = pd.Series(clf.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False).head(10))
