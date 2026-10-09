import pandas as pd
import re
import os

def parse_address(address):
    if not isinstance(address, str):
        return pd.Series({'pincode': None, 'landmark': None, 'house_no': None, 'street_info': None})
        
    address = address.lower()
    
    # 1. Extract Pincode (6 digits usually at the end)
    pincode_match = re.search(r'\b\d{6}\b', address)
    pincode = pincode_match.group(0) if pincode_match else None
    
    # 2. Extract Landmark
    landmark_keywords = ['near', 'behind', 'opp ', 'opposite', 'beside', 'next to', 'in front of', 'ke bagal mein', 'ke paas']
    landmark = None
    for kw in landmark_keywords:
        if kw in address:
            # extract text after the keyword until the next comma
            parts = address.split(kw)
            if len(parts) > 1:
                after_kw = parts[1].split(',')[0].strip()
                landmark = f"{kw} {after_kw}"
                break
                
    # 3. Extract House / Plot Number
    house_no_match = re.search(r'(?:no\.?|#|flat|plot|h\.?no\.?)\s*(\w+[\-\w]*)', address)
    house_no = house_no_match.group(1) if house_no_match else None
    
    # 4. Extract Street Info (Cross, Main, Road)
    street_match = re.search(r'(\d+(?:th|st|nd|rd)?\s+(?:cross|main|road))', address)
    street_info = street_match.group(1) if street_match else None
    
    return pd.Series({
        'pincode': pincode, 
        'landmark': landmark, 
        'house_no': house_no, 
        'street_info': street_info
    })

def main():
    data_dir_shared = r'c:\Users\kmzpa\Desktop\Open IIT Data\Dataset\data\shared'
    
    print("Loading addresses.csv...")
    addresses = pd.read_csv(os.path.join(data_dir_shared, 'addresses.csv'))
    
    print(f"Parsing {len(addresses)} addresses...")
    # Apply parser
    parsed_features = addresses['address_text'].apply(parse_address)
    
    # Combine with original
    result = pd.concat([addresses[['address_id', 'address_text']], parsed_features], axis=1)
    
    out_path = os.path.join(os.path.dirname(__file__), 'Extracted_Address_Features.csv')
    result.to_csv(out_path, index=False)
    print(f"Extraction complete! Saved to {out_path}")
    
    # Print some stats
    print("\n--- Parsing Statistics ---")
    print(f"Total Addresses: {len(result)}")
    print(f"Pincodes extracted: {result['pincode'].notna().sum()}")
    print(f"Landmarks extracted: {result['landmark'].notna().sum()}")
    print(f"House numbers extracted: {result['house_no'].notna().sum()}")
    print(f"Street Info extracted: {result['street_info'].notna().sum()}")
    
    # Print a few examples
    print("\n--- Examples ---")
    print(result[result['landmark'].notna()].head(3)[['address_text', 'landmark']].to_string())

if __name__ == "__main__":
    main()
