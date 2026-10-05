# %% [markdown]
# # PS1: Real-Time Fake PTP Detection (Demo Pipeline)
#
# Let's explore the Promise to Pay (PTP) data and build a basic predictive model.

# %% [markdown]
# ## 1. Imports & Setup
# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")
print("Libraries imported. Visuals configured.")

# %% [markdown]
# ## 2. Data Loading & Exploration
# Let's load the data and understand its structure.
# %%
base_path = r"../../Dataset/data"
ptps = pd.read_csv(os.path.join(base_path, "ps1_fake_ptp", "ptps.csv"))
accounts = pd.read_csv(os.path.join(base_path, "shared", "accounts.csv"))
payments = pd.read_csv(os.path.join(base_path, "shared", "payments.csv"))

print(f"Loaded {len(ptps)} PTPs.")

# %%
# Let's look at the promised amounts distribution
plt.figure(figsize=(8, 4))
sns.histplot(ptps['promised_amount'].dropna(), bins=50, kde=True)
plt.title("Distribution of Promised Amounts")
plt.xlabel("Promised Amount")
plt.ylabel("Frequency")
plt.show()

# %% [markdown]
# **Observation:** Most promised amounts are concentrated in the lower range, with a long tail of higher amounts.

# %% [markdown]
# ## 3. Data Preprocessing & Target Definition
# We need to determine if a PTP was actually kept by matching it with payments.
# %%
payments['payment_ts'] = pd.to_datetime(payments['payment_ts'])
ptps['captured_ts'] = pd.to_datetime(ptps['captured_ts'])
ptps['promised_date'] = pd.to_datetime(ptps['promised_date'])

ptp_payments = pd.merge(ptps, payments, on='account_id', how='left')
valid_payments = ptp_payments[
    (ptp_payments['payment_ts'] >= ptp_payments['captured_ts']) &
    (ptp_payments['payment_ts'] <= ptp_payments['promised_date'] + pd.Timedelta(days=3))
]

payment_sums = valid_payments.groupby('ptp_id')['amount'].sum().reset_index().rename(columns={'amount': 'total_paid'})
ptps = pd.merge(ptps, payment_sums, on='ptp_id', how='left').fillna({'total_paid': 0})

# Target: Kept if total paid >= 90% of promised amount
ptps['kept'] = (ptps['total_paid'] >= (ptps['promised_amount'] * 0.9)).astype(int)

# %%
plt.figure(figsize=(6, 4))
sns.countplot(x='kept', data=ptps, palette='pastel')
plt.title("PTP Kept vs Broken (Target Distribution)")
plt.show()

# %% [markdown]
# **Observation:** The dataset is imbalanced. Fake/Broken PTPs (0) are significantly more common than Kept PTPs (1).

# %% [markdown]
# ## 4. Feature Engineering
# Let's bring in account data and visualize relationships.
# %%
data = pd.merge(ptps, accounts, on='account_id', how='left')

features = ['promised_amount', 'overdue_at_capture', 'emi_amount', 'outstanding', 
            'ability_to_pay_estimate', 'prev_ptp_count', 'prev_ptp_broken']

for col in features:
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

# Let's see how outstanding amount relates to PTP being kept
plt.figure(figsize=(8, 5))
sns.boxplot(x='kept', y='outstanding', data=data)
plt.title("Outstanding Amount vs PTP Kept")
plt.yscale('log')
plt.show()

# %% [markdown]
# ## 5. Modeling
# %%
X = data[features]
y = data['kept']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print("\n--- Model Evaluation ---")
print(classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
