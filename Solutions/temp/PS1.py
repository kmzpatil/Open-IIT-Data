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
from sklearn.metrics import classification_report, roc_auc_score
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
# ## 4. Deep NLP Feature Engineering (Transcripts)
# %%
def extract_nlp_features(df):
    cust_turns = df[df['speaker'] == 'customer']
    cust_text = cust_turns['text'].astype(str).str.lower()
    full_text = ' '.join(cust_text)
    
    hesitation_markers = ['dekhta', 'try', 'maybe', 'not sure', 'kashta', 'gottilla', 'will see', 'probably', 'koshish']
    condition_markers = ['salary', 'if', 'aayi toh', 'bandre', 'when', 'after']
    
    hesitation_count = sum(full_text.count(marker) for marker in hesitation_markers)
    condition_count = sum(full_text.count(marker) for marker in condition_markers)
    word_count = len(full_text.split())
    
    avg_pause = pd.to_numeric(cust_turns['pause_before_ms'], errors='coerce').mean()
    max_pause = pd.to_numeric(cust_turns['pause_before_ms'], errors='coerce').max()
    avg_asr = pd.to_numeric(cust_turns['asr_confidence'], errors='coerce').mean()
    num_turns = len(cust_turns)
    
    return pd.Series({
        'hesitation_count': hesitation_count, 
        'condition_count': condition_count,
        'word_count': word_count,
        'avg_pause_ms': avg_pause if pd.notnull(avg_pause) else 0,
        'max_pause_ms': max_pause if pd.notnull(max_pause) else 0,
        'avg_asr_conf': avg_asr if pd.notnull(avg_asr) else 1.0,
        'num_turns': num_turns
    })

transcript_features = transcripts.groupby('call_id').apply(extract_nlp_features).reset_index()
transcript_features.rename(columns={'call_id': 'attempt_id'}, inplace=True)

data = pd.merge(ptps, accounts, on='account_id', how='left')
data = pd.merge(data, transcript_features, on='attempt_id', how='left').fillna({
    'hesitation_count':0, 'condition_count':0, 'word_count':0, 
    'avg_pause_ms':0, 'max_pause_ms':0, 'avg_asr_conf':1.0, 'num_turns':0
})

# %%
plt.figure(figsize=(8, 5))
sns.kdeplot(data=data, x='avg_pause_ms', hue='kept', fill=True, common_norm=False)
plt.title("Customer Average Pause Duration vs PTP Kept")
plt.xlim(0, 5000)
plt.show()

# %% [markdown]
# **Observation:** Longer average pauses in speech are somewhat indicative of broken PTPs.

# %% [markdown]
# ## 5. Advanced Feature Engineering
# %%
# Temporal Features
data['time_to_promise_days'] = (data['promised_date'] - data['captured_ts']).dt.days
data['promise_day_of_month'] = data['promised_date'].dt.day
data['is_close_to_salary'] = (np.abs(data['promise_day_of_month'] - data['salary_credit_day']) <= 3).astype(int)

# Ratio Features
data['promise_to_overdue_ratio'] = data['promised_amount'] / (data['overdue_at_capture'] + 1e-5)
data['historical_break_rate'] = data['prev_ptp_broken'] / (data['prev_ptp_count'] + 1e-5)

# Bureau Score Encoding
bureau_mapping = {'<550': 1, '550-649': 2, '650-699': 3, '700-749': 4, '750-799': 5, '>800': 6, 'X': 0}
data['bureau_score_encoded'] = data['bureau_score_band'].map(bureau_mapping).fillna(0)

num_features = ['promised_amount', 'overdue_at_capture', 'emi_amount', 'outstanding', 
                'ability_to_pay_estimate', 'prev_ptp_count', 'prev_ptp_broken', 
                'hesitation_count', 'condition_count', 'word_count', 
                'avg_pause_ms', 'max_pause_ms', 'avg_asr_conf', 'num_turns',
                'time_to_promise_days', 'promise_to_overdue_ratio', 'historical_break_rate',
                'bureau_score_encoded', 'is_close_to_salary']

for col in num_features:
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

cat_features = ['channel', 'income_type']
data = pd.get_dummies(data, columns=cat_features, drop_first=True)

# %%
plt.figure(figsize=(10, 8))
corr = data[['kept', 'time_to_promise_days', 'promise_to_overdue_ratio', 'bureau_score_encoded', 'hesitation_count', 'avg_pause_ms']].corr()
sns.heatmap(corr, annot=True, cmap='RdYlGn', center=0)
plt.title("Correlation of Advanced Features with Target")
plt.show()

# %% [markdown]
# ## 6. Modeling & Evaluation
# %%
features = num_features + [c for c in data.columns if c.startswith('channel_') or c.startswith('income_type_')]
X = data[features]
y = data['kept']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Highly tuned Gradient Boosting
clf = HistGradientBoostingClassifier(
    max_iter=500, learning_rate=0.03, max_depth=8, 
    l2_regularization=0.1, random_state=42, class_weight='balanced'
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
