from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os

app = FastAPI(title="Geocoding_Engine Address Geocoder API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "Geocoding_Engine_Output.csv")
df = None

@app.on_event("startup")
def load_data():
    global df
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df.fillna("", inplace=True)
    else:
        df = pd.DataFrame()

@app.get("/")
def read_root():
    return {"message": "Welcome to Geocoding_Engine Geocoder API"}

@app.get("/api/search")
def search_address(q: str = Query(..., min_length=2)):
    if df.empty:
        return {"error": "Data not loaded"}
    
    q_lower = q.lower()
    
    # Substring match for ID
    id_mask = df["address_id"].str.contains(q, case=False, na=False)
    id_indices = df[id_mask].index.tolist()
    
    # Fuzzy match for address text
    from rapidfuzz import process, fuzz
    choices = df["address_text"].tolist()
    
    # extract top matches using partial_ratio to handle substrings with typos
    matches = process.extract(q_lower, choices, scorer=fuzz.partial_ratio, limit=50)
    
    # Keep matches with a score above 65
    fuzzy_indices = [match[2] for match in matches if match[1] > 65]
    
    # Combine and remove duplicates, preserving some order
    final_indices = list(dict.fromkeys(id_indices + fuzzy_indices))
    
    results = df.iloc[final_indices][:50].to_dict(orient="records")
    return {"results": results}

@app.get("/api/stats")
def get_stats():
    if df.empty:
        return {"error": "Data not loaded"}
    
    tier_counts = df["tier"].value_counts().to_dict()
    total = len(df)
    return {
        "total_addresses": total,
        "tiers": tier_counts
    }
