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
from sklearn.metrics import classification_report, roc_auc_score
import os

sns.set_theme(style="darkgrid", palette="deep")
print("Environment initialized.")

base_path = r"../../Dataset/data"
phones = pd.read_csv(os.path.join(base_path, "ps2_right_party_contact", "phones.csv"))
dial_attempts = pd.read_csv(os.path.join(base_path, "shared", "dial_attempts.csv"))
verified = pd.read_csv(os.path.join(base_path, "ps2_right_party_contact", "verified_contact_points.csv"))

# %% [markdown]
# ## 2. Advanced Feature Engineering
# %%
verified['is_rpc'] = (verified['verified_status'] == 'borrower_number').astype(int)
dial_attempts['attempt_ts'] = pd.to_datetime(dial_attempts['attempt_ts'])

anchor_date = pd.to_datetime('2026-08-01')

# Extract detailed network responses and connection features
attempts_agg = dial_attempts.groupby('phone_id').agg(
    total_attempts=('attempt_id', 'count'),
    switched_off_count=('network_response', lambda x: (x == 'switched_off').sum()),
    ringing_count=('network_response', lambda x: (x == 'ringing').sum()),
    invalid_count=('network_response', lambda x: (x == 'invalid_number').sum()),
    answered_count=('talk_duration_s', lambda x: (x > 0).sum()),
    last_attempt_date=('attempt_ts', 'max'),
    avg_talk_time=('talk_duration_s', 'mean'),
    max_talk_time=('talk_duration_s', 'max'),
    avg_ring_time=('ring_duration_s', 'mean')
).reset_index()

attempts_agg['switched_off_ratio'] = attempts_agg['switched_off_count'] / attempts_agg['total_attempts']
attempts_agg['ringing_ratio'] = attempts_agg['ringing_count'] / attempts_agg['total_attempts']
attempts_agg['invalid_ratio'] = attempts_agg['invalid_count'] / attempts_agg['total_attempts']
attempts_agg['answered_ratio'] = attempts_agg['answered_count'] / attempts_agg['total_attempts']
attempts_agg['days_since_last_attempt'] = (anchor_date - attempts_agg['last_attempt_date']).dt.days

data = pd.merge(verified, phones, on='phone_id', how='left')
data = pd.merge(data, attempts_agg, on='phone_id', how='left').fillna(
    {'total_attempts':0, 'switched_off_ratio':0, 'ringing_ratio':0, 'invalid_ratio':0, 
     'answered_ratio':0, 'days_since_last_attempt':999, 'avg_talk_time': 0, 
     'max_talk_time': 0, 'avg_ring_time': 0}
)

# Encode categorical Phone metadata
data = pd.get_dummies(data, columns=['source', 'relation_recorded'], drop_first=True)

# %%
# Visualizing Network Response Ratios vs RPC
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
sns.boxplot(x='is_rpc', y='switched_off_ratio', data=data, ax=axs[0])
axs[0].set_title("Switched Off Ratio vs RPC")

sns.boxplot(x='is_rpc', y='answered_ratio', data=data, ax=axs[1])
axs[1].set_title("Answered Ratio vs RPC")

sns.boxplot(x='is_rpc', y='invalid_ratio', data=data, ax=axs[2])
axs[2].set_title("Invalid Ratio vs RPC")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Observation:** Phones that are frequently "switched off" or "invalid" have practically 0% chance of being an RPC. High answer rates strongly correlate with actual borrower numbers.

# %% [markdown]
# ## 3. Modeling
# %%
num_features = ['priority_slot', 'total_attempts', 'switched_off_ratio', 'ringing_ratio', 
                'invalid_ratio', 'answered_ratio', 'days_since_last_attempt', 
                'avg_talk_time', 'max_talk_time', 'avg_ring_time']
cat_features = [c for c in data.columns if c.startswith('source_') or c.startswith('relation_')]

features = num_features + cat_features
X = data[features]
y = data['is_rpc']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.03, max_depth=7, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
