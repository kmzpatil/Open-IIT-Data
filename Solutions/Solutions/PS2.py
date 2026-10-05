# %% [markdown]
# # PS2: Right-Party Contact Prediction (Demo Pipeline)
# Let's explore the dialing data and predict Right-Party Contacts.

# %% [markdown]
# ## 1. Setup & Data Loading
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import os

sns.set_theme(style="whitegrid")

base_path = r"../../Dataset/data"
phones = pd.read_csv(os.path.join(base_path, "ps2_right_party_contact", "phones.csv"))
dial_attempts = pd.read_csv(os.path.join(base_path, "shared", "dial_attempts.csv"))
verified = pd.read_csv(os.path.join(base_path, "ps2_right_party_contact", "verified_contact_points.csv"))

# %% [markdown]
# ## 2. Data Exploration
# %%
# Let's see the network responses we get when calling
plt.figure(figsize=(10, 5))
sns.countplot(y='network_response', data=dial_attempts, order=dial_attempts['network_response'].value_counts().index)
plt.title("Distribution of Network Responses")
plt.show()

# %% [markdown]
# **Observation:** "Ringing" and "Switched Off" are the most common network responses.

# %% [markdown]
# ## 3. Feature Engineering
# %%
verified['is_rpc'] = (verified['verified_status'] == 'borrower_number').astype(int)

attempts_agg = dial_attempts.groupby('phone_id').agg(
    total_attempts=('attempt_id', 'count'),
    avg_ring_duration=('ring_duration_s', 'mean'),
    total_talk_time=('talk_duration_s', 'sum')
).reset_index()

data = pd.merge(verified, phones, on='phone_id', how='left')
data = pd.merge(data, attempts_agg, on='phone_id', how='left').fillna({'total_attempts': 0, 'avg_ring_duration': 0, 'total_talk_time': 0})

# %%
# Let's plot Total Talk Time against our RPC target
plt.figure(figsize=(8, 5))
sns.kdeplot(data=data, x='total_talk_time', hue='is_rpc', fill=True, common_norm=False)
plt.title("Density of Total Talk Time by Right-Party Contact")
plt.xlim(0, 200)
plt.show()

# %% [markdown]
# ## 4. Modeling
# %%
features = ['priority_slot', 'total_attempts', 'avg_ring_duration', 'total_talk_time']
X = data[features]
y = data['is_rpc']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=3)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
