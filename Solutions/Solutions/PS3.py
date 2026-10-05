# %% [markdown]
# # PS3: Address Geocoder (Demo Pipeline)
# Simple geocoding baseline using field visits.

# %% [markdown]
# ## 1. Setup
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")

base_path = r"../../Dataset/data"
addresses = pd.read_csv(os.path.join(base_path, "shared", "addresses.csv"))
baseline = pd.read_csv(os.path.join(base_path, "ps3_geocoder", "baseline_geocodes.csv"))
field_visits = pd.read_csv(os.path.join(base_path, "shared", "field_visits.csv"))
surveyed = pd.read_csv(os.path.join(base_path, "ps3_geocoder", "surveyed_addresses.csv"))

# %% [markdown]
# ## 2. EDA: Field Visit Outcomes
# %%
plt.figure(figsize=(10, 5))
sns.countplot(y='outcome', data=field_visits, order=field_visits['outcome'].value_counts().index)
plt.title("Outcomes of Field Visits")
plt.show()

# %% [markdown]
# **Observation:** We only want to use visits that successfully located the address (e.g. met_borrower).

# %% [markdown]
# ## 3. Processing
# %%
success_outcomes = ['cash_collected', 'ptp_collected', 'met_borrower']
successful_visits = field_visits[field_visits['outcome'].isin(success_outcomes)]

address_gps = successful_visits.groupby('address_id').agg(
    visit_x=('checkin_x', 'mean'),
    visit_y=('checkin_y', 'mean')
).reset_index()

data = pd.merge(addresses, baseline, on='address_id', how='left')
data = pd.merge(data, address_gps, on='address_id', how='left')

data['pred_x'] = data['visit_x'].fillna(data['geocoder_x'])
data['pred_y'] = data['visit_y'].fillna(data['geocoder_y'])

# %% [markdown]
# ## 4. Evaluation
# %%
eval_data = pd.merge(surveyed, data, on='address_id', how='inner')
eval_data['model_error'] = np.sqrt((eval_data['pred_x'] - eval_data['surveyed_x'])**2 + (eval_data['pred_y'] - eval_data['surveyed_y'])**2)

print(f"Median Error: {eval_data['model_error'].median():.2f}")
print(f"90th Percentile: {eval_data['model_error'].quantile(0.90):.2f}")
