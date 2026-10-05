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
from sklearn.cluster import DBSCAN

sns.set_theme(style="darkgrid")
print("Environment initialized.")

base_path = r"../../Dataset/data"
addresses = pd.read_csv(os.path.join(base_path, "shared", "addresses.csv"))
baseline = pd.read_csv(os.path.join(base_path, "ps3_geocoder", "baseline_geocodes.csv"))
field_visits = pd.read_csv(os.path.join(base_path, "shared", "field_visits.csv"))
surveyed = pd.read_csv(os.path.join(base_path, "ps3_geocoder", "surveyed_addresses.csv"))

# %% [markdown]
# ## 2. High-Integrity Mathematical Filtering & Clustering
# %%
# We strictly consider visits that verified the location
valid_outcomes = ['met_borrower', 'met_family', 'door_locked']
valid_visits = field_visits[
    (field_visits['outcome'].isin(valid_outcomes)) &
    (field_visits['gps_accuracy_m'] <= 20) &  # Stricter GPS accuracy
    (field_visits['dwell_s'] >= 120)          # Stricter Dwell Time (minimum 2 mins)
]

def calculate_centroid_dbscan(df):
    if len(df) >= 3:
        # Use DBSCAN for spatial clustering to discard GPS anomalies
        coords = df[['checkin_x', 'checkin_y']].values
        clustering = DBSCAN(eps=20, min_samples=2).fit(coords)
        df = df.copy()
        df['cluster'] = clustering.labels_
        
        # Keep points in the largest cluster (or noise if none found)
        largest_cluster = df['cluster'].mode()[0]
        if largest_cluster != -1:
            df = df[df['cluster'] == largest_cluster]

    # Inverse Variance Weighting
    weights = 1.0 / (df['gps_accuracy_m'] + 1e-5)
    mean_x = np.average(df['checkin_x'], weights=weights)
    mean_y = np.average(df['checkin_y'], weights=weights)
    
    return pd.Series({'visit_x': mean_x, 'visit_y': mean_y})

address_gps = valid_visits.groupby('address_id').apply(calculate_centroid_dbscan).reset_index()

data = pd.merge(addresses, baseline, on='address_id', how='left')
data = pd.merge(data, address_gps, on='address_id', how='left')

# Fall back to baseline geocoder if no valid high-integrity visit was found
data['pred_x'] = data['visit_x'].fillna(data['geocoder_x'])
data['pred_y'] = data['visit_y'].fillna(data['geocoder_y'])

# %% [markdown]
# ## 3. Evaluation
# %%
eval_data = pd.merge(surveyed, data, on='address_id', how='inner')
eval_data['model_error'] = np.sqrt((eval_data['pred_x'] - eval_data['surveyed_x'])**2 + (eval_data['pred_y'] - eval_data['surveyed_y'])**2)
eval_data['baseline_error'] = np.sqrt((eval_data['geocoder_x'] - eval_data['surveyed_x'])**2 + (eval_data['geocoder_y'] - eval_data['surveyed_y'])**2)

# %%
# Visualizing Error Improvement
plt.figure(figsize=(10, 5))
sns.kdeplot(eval_data['baseline_error'], label='Baseline Geocoder', fill=True, color='red', alpha=0.3)
sns.kdeplot(eval_data['model_error'], label='Advanced Clustering Model', fill=True, color='blue', alpha=0.3)
plt.title("Error Distribution: Baseline vs Advanced Model")
plt.xlabel("Error (meters)")
plt.xlim(0, 500)
plt.legend()
plt.show()

# %%
print(f"Baseline Median Error: {eval_data['baseline_error'].median():.2f}")
print(f"Advanced Median Error: {eval_data['model_error'].median():.2f}")
print(f"Advanced 90th Percentile: {eval_data['model_error'].quantile(0.90):.2f}")
