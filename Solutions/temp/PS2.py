# %% [markdown]
# # PS2: Right-Party Contact Prediction (Advanced EDA & Pipeline)
# 
# We'll dive deep into dialing history, looking at time-decay features and connection ratios to predict valid phone numbers.

# %% [markdown]
# ## 1. Setup & Imports
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import os

sns.set_theme(style="darkgrid", palette="deep")

base_path = r"../../Dataset/data"
phones = pd.read_csv(os.path.join(base_path, "ps2_right_party_contact", "phones.csv"))
dial_attempts = pd.read_csv(os.path.join(base_path, "shared", "dial_attempts.csv"))
verified = pd.read_csv(os.path.join(base_path, "ps2_right_party_contact", "verified_contact_points.csv"))

# %% [markdown]
# ## 2. Deep Data Exploration
# %%
# Exploring talk duration distribution
plt.figure(figsize=(10, 5))
sns.histplot(dial_attempts[dial_attempts['talk_duration_s'] > 0]['talk_duration_s'], bins=50, kde=True, color='purple')
plt.title("Distribution of Talk Duration (for connected calls)")
plt.xlabel("Seconds")
plt.show()

# %% [markdown]
# **Inference:** The majority of calls are short (under 30 seconds). Very long calls are rare.

# %% [markdown]
# ## 3. Advanced Feature Engineering (Time-Decay)
# %%
verified['is_rpc'] = (verified['verified_status'] == 'borrower_number').astype(int)
dial_attempts['attempt_ts'] = pd.to_datetime(dial_attempts['attempt_ts'])

dial_attempts = dial_attempts.sort_values('attempt_ts')
anchor_date = pd.to_datetime('2026-08-01')

# Extract detailed network responses
attempts_agg = dial_attempts.groupby('phone_id').agg(
    total_attempts=('attempt_id', 'count'),
    switched_off_count=('network_response', lambda x: (x == 'switched_off').sum()),
    answered_count=('talk_duration_s', lambda x: (x > 0).sum()),
    last_attempt_date=('attempt_ts', 'max'),
    avg_talk_time=('talk_duration_s', 'mean')
).reset_index()

attempts_agg['switched_off_ratio'] = attempts_agg['switched_off_count'] / attempts_agg['total_attempts']
attempts_agg['answered_ratio'] = attempts_agg['answered_count'] / attempts_agg['total_attempts']
attempts_agg['days_since_last_attempt'] = (anchor_date - attempts_agg['last_attempt_date']).dt.days

data = pd.merge(verified, phones, on='phone_id', how='left')
data = pd.merge(data, attempts_agg, on='phone_id', how='left').fillna(
    {'total_attempts':0, 'switched_off_ratio':0, 'answered_ratio':0, 'days_since_last_attempt':-1, 'avg_talk_time': 0}
)

# %%
# Correlation of new features
plt.figure(figsize=(10, 6))
corr = data[['is_rpc', 'switched_off_ratio', 'answered_ratio', 'days_since_last_attempt', 'avg_talk_time']].corr()
sns.heatmap(corr, annot=True, cmap='RdBu', center=0)
plt.title("Correlation Matrix for RPC Features")
plt.show()

# %% [markdown]
# **Observation:** `answered_ratio` and `avg_talk_time` are highly positively correlated with Right-Party Contact, while `switched_off_ratio` is negatively correlated.

# %% [markdown]
# ## 4. Modeling
# %%
features = ['total_attempts', 'switched_off_ratio', 'answered_ratio', 'days_since_last_attempt', 'avg_talk_time', 'priority_slot']
X = data[features]
y = data['is_rpc']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
