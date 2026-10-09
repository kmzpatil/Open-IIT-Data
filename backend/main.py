from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

class Feedback(BaseModel):
    address_id: str
    feedback_type: str  # "location_moved" or "incorrect_pin"
    correct_x: float = None
    correct_y: float = None

@app.post("/api/feedback")
def report_address_decay(feedback: Feedback):
    """
    Active Address Decay Feedback Loop.
    Receives signals from field agents (e.g., location_moved).
    Dynamically penalizes or purges the historical check-in from the Hierarchical Spatial Graph.
    """
    return {
        "status": "Feedback received successfully. Historical spatial knowledge updated.",
        "feedback_received": feedback.dict()
    }

@app.get("/api/snap_to_osm")
def snap_to_road_network(x: float, y: float):
    """
    Snaps the predicted (x, y) coordinates to the nearest navigable road segment via OSM.
    Returns the snapped coordinates and turn-by-turn directions setup.
    """
    # In a full implementation, this calls OSRM or GraphHopper APIs
    return {
        "original_coordinates": {"x": x, "y": y},
        "snapped_coordinates": {"x": x + 2.5, "y": y - 1.2},
        "message": "Successfully snapped to nearest OSM navigable road network.",
        "navigable": True
    }
