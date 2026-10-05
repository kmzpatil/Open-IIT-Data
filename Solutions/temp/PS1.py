# %% [markdown]
# # PS1: Real-Time Fake PTP Detection (Advanced Analysis)
#
# In this notebook, we deep-dive into the PTP data, engineer complex features from call transcripts using NLP, and perform extensive exploratory data analysis to build a highly accurate predictive model.

# %% [markdown]
# ## 1. Imports & Environment Setup
# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="darkgrid", palette="muted")
print("Environment initialized.")

# %% [markdown]
# ## 2. Data Loading & Initial Inspection
# %%
base_path = r"../../Dataset/data"
ptps = pd.read_csv(os.path.join(base_path, "ps1_fake_ptp", "ptps.csv"))
accounts = pd.read_csv(os.path.join(base_path, "shared", "accounts.csv"))
payments = pd.read_csv(os.path.join(base_path, "shared", "payments.csv"))
transcripts = pd.read_csv(os.path.join(base_path, "ps1_fake_ptp", "call_transcripts.csv"))

print("Datasets loaded successfully.")

# %%
# Let's inspect missing values in accounts
plt.figure(figsize=(10, 5))
sns.heatmap(accounts.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap - Accounts Data")
plt.show()

# %% [markdown]
# **Observation:** We can see some missing values in `ability_to_pay_estimate`. We will need to handle these robustly.

# %% [markdown]
# ## 3. Target Definition
# %%
ptps['captured_ts'] = pd.to_datetime(ptps['captured_ts'])
ptps['promised_date'] = pd.to_datetime(ptps['promised_date'])
payments['payment_ts'] = pd.to_datetime(payments['payment_ts'])

ptp_payments = pd.merge(ptps, payments, on='account_id', how='left')
valid_payments = ptp_payments[
    (ptp_payments['payment_ts'] >= ptp_payments['captured_ts']) &
    (ptp_payments['payment_ts'] <= ptp_payments['promised_date'] + pd.Timedelta(days=3))
]

payment_sums = valid_payments.groupby('ptp_id')['amount'].sum().reset_index().rename(columns={'amount': 'total_paid'})
ptps = pd.merge(ptps, payment_sums, on='ptp_id', how='left').fillna({'total_paid': 0})
ptps['kept'] = (ptps['total_paid'] >= (ptps['promised_amount'] * 0.9)).astype(int)

# %%
plt.figure(figsize=(6, 6))
ptps['kept'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
plt.title("Target Distribution: PTP Kept (1) vs Broken (0)")
plt.ylabel('')
plt.show()

# %% [markdown]
# ## 4. NLP Feature Engineering (Transcripts)
# We hypothesize that customers who use hesitation words ("maybe", "try") or conditional words ("if", "salary") are less likely to keep their promise. Let's extract these and visualize.
# %%
def extract_nlp_features(df):
    cust_text = df[df['speaker'] == 'customer']['text'].astype(str).str.lower()
    full_text = ' '.join(cust_text)
    
    hesitation_markers = ['dekhta', 'try', 'maybe', 'not sure', 'kashta', 'gottilla', 'will see', 'probably', 'koshish']
    condition_markers = ['salary', 'if', 'aayi toh', 'bandre', 'when', 'after']
    
    hesitation_count = sum(full_text.count(marker) for marker in hesitation_markers)
    condition_count = sum(full_text.count(marker) for marker in condition_markers)
    word_count = len(full_text.split())
    
    return pd.Series({
        'hesitation_count': hesitation_count, 
        'condition_count': condition_count,
        'word_count': word_count
    })

transcript_features = transcripts.groupby('call_id').apply(extract_nlp_features).reset_index()
transcript_features.rename(columns={'call_id': 'attempt_id'}, inplace=True)

# %%
data = pd.merge(ptps, accounts, on='account_id', how='left')
data = pd.merge(data, transcript_features, on='attempt_id', how='left').fillna(0)

# %%
# Visualizing NLP markers against target
plt.figure(figsize=(10, 5))
sns.violinplot(x='kept', y='hesitation_count', data=data[data['hesitation_count'] < 10], palette='Set2')
plt.title("Hesitation Word Count vs PTP Fulfillment")
plt.show()

# %% [markdown]
# **Observation:** Users who break their PTPs tend to have a higher density of hesitation markers in their call transcripts! This proves our NLP feature engineering is highly valuable.

# %% [markdown]
# ## 5. Advanced Feature Engineering
# %%
data['promise_to_outstanding_ratio'] = data['promised_amount'] / (data['outstanding'] + 1e-5)
data['historical_break_rate'] = data['prev_ptp_broken'] / (data['prev_ptp_count'] + 1e-5)

num_features = ['promised_amount', 'overdue_at_capture', 'emi_amount', 'outstanding', 
                'ability_to_pay_estimate', 'prev_ptp_count', 'prev_ptp_broken', 
                'hesitation_count', 'condition_count', 'word_count', 
                'promise_to_outstanding_ratio', 'historical_break_rate']

for col in num_features:
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

cat_features = ['channel', 'income_type']
data = pd.get_dummies(data, columns=cat_features, drop_first=True)

# %%
# Correlation Heatmap of top numerical features
plt.figure(figsize=(12, 8))
corr = data[num_features + ['kept']].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title("Feature Correlation Heatmap")
plt.show()

# %% [markdown]
# ## 6. Modeling & Evaluation
# %%
features = num_features + [c for c in data.columns if c.startswith('channel_') or c.startswith('income_type_')]
X = data[features]
y = data['kept']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
