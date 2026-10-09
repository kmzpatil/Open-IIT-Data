import pandas as pd
import os

data_dir_shared = os.path.join('data', 'raw', 'Dataset', 'data', 'shared')
data_dir_engine = os.path.join('data', 'raw', 'Dataset', 'data', 'address_geocoder')

addresses = pd.read_csv(os.path.join(data_dir_shared, 'addresses.csv'))
localities = pd.read_csv(os.path.join(data_dir_engine, 'localities.csv'))
landmarks = pd.read_csv(os.path.join(data_dir_engine, 'landmarks_poi.csv'))

# lowercase
localities['locality_name'] = localities['locality_name'].str.lower()
landmarks['name'] = landmarks['name'].str.lower()
addresses['text_low'] = addresses['address_text'].str.lower()

loc_matches = 0
lm_matches = 0

for idx, row in addresses.iterrows():
    text = str(row['text_low'])
    
    # Check localities
    t_id = row['town_id']
    town_locs = localities[localities['town_id'] == t_id]
    for _, l_row in town_locs.iterrows():
        if l_row['locality_name'] in text:
            loc_matches += 1
            break
            
    town_lms = landmarks[landmarks['town_id'] == t_id]
    for _, lm_row in town_lms.iterrows():
        if lm_row['name'] in text:
            lm_matches += 1
            break

print(f"Total addresses: {len(addresses)}")
print(f"Locality matches: {loc_matches}")
print(f"Landmark matches: {lm_matches}")
