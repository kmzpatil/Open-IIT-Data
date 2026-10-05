# %% [markdown]
# # PS3: Address Geocoder (Advanced GPS Telemetry & Analysis)
# Advanced mathematical modeling of field visit GPS telemetry to pinpoint addresses.

# %% [markdown]
# ## 1. Setup & Imports
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="darkgrid")

base_path = r"../../Dataset/data"
addresses = pd.read_csv(os.path.join(base_path, "shared", "addresses.csv"))
baseline = pd.read_csv(os.path.join(base_path, "ps3_geocoder", "baseline_geocodes.csv"))
field_visits = pd.read_csv(os.path.join(base_path, "shared", "field_visits.csv"))
surveyed = pd.read_csv(os.path.join(base_path, "ps3_geocoder", "surveyed_addresses.csv"))

# %% [markdown]
# ## 2. Data Exploration: GPS Accuracy & Dwell Time
# %%
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(field_visits['gps_accuracy_m'], bins=50, ax=axs[0], color='orange')
axs[0].set_title("Distribution of GPS Accuracy (m)")
axs[0].set_xlim(0, 100)

sns.histplot(field_visits['dwell_s'], bins=50, ax=axs[1], color='green')
axs[1].set_title("Distribution of Dwell Time (s)")
axs[1].set_xlim(0, 1000)

plt.show()

# %% [markdown]
# **Observation:** A significant number of visits have poor GPS accuracy or very low dwell times (meaning the agent was just passing by). We must filter these out to build an accurate geocoder.

# %% [markdown]
# ## 3. High-Integrity Mathematical Filtering
# %%
valid_outcomes = ['met_borrower', 'met_family', 'cash_collected', 'door_locked']
valid_visits = field_visits[
    (field_visits['outcome'].isin(valid_outcomes)) &
    (field_visits['gps_accuracy_m'] <= 25) & 
    (field_visits['dwell_s'] >= 60)
]

def calculate_centroid_and_confidence(df):
    weights = 1.0 / (df['gps_accuracy_m'] + 1e-5)
    mean_x = np.average(df['checkin_x'], weights=weights)
    mean_y = np.average(df['checkin_y'], weights=weights)
    
    var_x = np.average((df['checkin_x'] - mean_x)**2, weights=weights)
    var_y = np.average((df['checkin_y'] - mean_y)**2, weights=weights)
    
    radius = np.sqrt(var_x + var_y) + df['gps_accuracy_m'].mean()
    return pd.Series({'visit_x': mean_x, 'visit_y': mean_y, 'conf_radius': radius})

address_gps = valid_visits.groupby('address_id').apply(calculate_centroid_and_confidence).reset_index()

data = pd.merge(addresses, baseline, on='address_id', how='left')
data = pd.merge(data, address_gps, on='address_id', how='left')
data['pred_x'] = data['visit_x'].fillna(data['geocoder_x'])
data['pred_y'] = data['visit_y'].fillna(data['geocoder_y'])

# %% [markdown]
# ## 4. Evaluation
# %%
eval_data = pd.merge(surveyed, data, on='address_id', how='inner')
eval_data['model_error'] = np.sqrt((eval_data['pred_x'] - eval_data['surveyed_x'])**2 + (eval_data['pred_y'] - eval_data['surveyed_y'])**2)

# %%
# Visualizing Error Improvement
eval_data['baseline_error'] = np.sqrt((eval_data['geocoder_x'] - eval_data['surveyed_x'])**2 + (eval_data['geocoder_y'] - eval_data['surveyed_y'])**2)

plt.figure(figsize=(10, 5))
sns.kdeplot(eval_data['baseline_error'], label='Baseline Geocoder', fill=True, color='red', alpha=0.3)
sns.kdeplot(eval_data['model_error'], label='Advanced GPS Model', fill=True, color='blue', alpha=0.3)
plt.title("Error Distribution: Baseline vs Advanced Model")
plt.xlabel("Error (meters)")
plt.xlim(0, 200)
plt.legend()
plt.show()

# %%
print(f"Median Error: {eval_data['model_error'].median():.2f}")
print(f"90th Percentile: {eval_data['model_error'].quantile(0.90):.2f}")
