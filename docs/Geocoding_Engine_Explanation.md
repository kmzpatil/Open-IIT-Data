# Geocoding Engine: Explanatory Document

## 1. Problem Statement (PS)

**Objective**: 
Build a robust spatial resolution engine capable of geocoding unstructured, noisy addresses into precise geographical coordinates (x, y). The system must learn from historical field agent visits and correct the errors of a baseline (often inaccurate) geocoding service.

- **Input**: 
  - Raw address strings (e.g., "NO 123 BEHIND HANUMAN MANDIR GANDHI NAGAR").
  - Metadata: `address_id`, `town_id`, `address_type`, and `pincode`.
  
- **Output**: 
  - Verified coordinates (`x`, `y` in local meters).
  - Confidence radius (in meters, representing spatial uncertainty).
  - Human-readable directions (e.g., "120m SW of Ganesh Temple").
  - Verification Tier (`A`, `B`, or `C`).

---

## 2. Database (Data Sources & Usage)

The engine relies on a fusion of text, spatial, and behavioral data to overcome inaccurate baseline geocodes.

1. **`addresses.csv`**:
   - **What it is**: The core list of addresses to be geocoded (contains the raw `address_text`, `town_id`, etc.).
   - **How we use it**: Base input for the entire pipeline. The text is parsed for structural elements and embedded into vector space for machine learning.
   
2. **`baseline_geocodes.csv`**:
   - **What it is**: Initial estimates from a standard geocoding API (e.g., Google Maps). Often inaccurate in rural/unstructured areas.
   - **How we use it**: Serves as the starting anchor (`geocoder_x`, `geocoder_y`). Our ML model predicts the *residual error* (delta) of these baseline coordinates rather than predicting absolute coordinates from scratch.
   
3. **`field_visits.csv` & `visit_gps_points.csv`**:
   - **What it is**: Historical data from field agents (collections/verification). Includes exact timestamps, outcomes, and GPS coordinates of where they checked in.
   - **How we use it**: Used to establish **Ground Truth**. We filter out fraudulent visits (GPS spoofing, duplicated photos), and the valid check-ins become the absolute truth our model trains on.
   
4. **`localities.csv` & `landmarks_poi.csv`**:
   - **What it is**: Verified coordinates of known neighborhoods (localities) and Points of Interest (landmarks/temples/bus stops).
   - **How we use it**: Provides **Explicit Spatial Anchors**. If an address mentions "Hanuman Mandir", we extract the exact coordinates of that temple and feed them as highly-weighted features into the ML model. It is also used to generate the final human-readable directions.

---

## 3. Pipeline Architecture

The pipeline processes addresses in 5 distinct stages.

### Stage 1: Address Parsing
- **Goal**: Extract structured components from noisy text.
- **Input**: Raw `address_text`.
- **Output**: `parsed_pincode`, `parsed_house_no`.
- **Example**:
  - *Input*: `"FLAT 4B OPPOSITE RATION SHOP 960102"`
  - *Output*: House No: `4B`, Pincode: `960102`.

### Stage 2: Fraud Detection & Ground Truth Extraction
- **Goal**: Determine absolute true locations based on successful field visits while ignoring spoofed/fraudulent agent activity.
- **Input**: `field_visits.csv`, `baseline_geocodes.csv`.
- **Output**: Verified median coordinates (`checkin_x`, `checkin_y`) for visited addresses.
- **Example**: 
  - *Input*: Agent logs 3 visits to Address `AD001`. One visit is 2000m away from the baseline (flagged as spoofing). Two visits are clustered at `(100, 200)`.
  - *Output*: Ground truth for `AD001` is set to `(100, 200)`.

### Stage 3: Explicit Spatial Anchoring
- **Goal**: Find known spatial anchors mentioned in the address text.
- **Input**: `address_text`, `localities.csv`, `landmarks_poi.csv`.
- **Output**: Extracted coordinates (`loc_x`, `loc_y`, `lm_x`, `lm_y`).
- **Example**:
  - *Input*: `"Near Ganesh Temple, Vinayaka Nagar"`
  - *Output*: `loc_x, loc_y` = `(-2863.9, 1096.5)` (Vinayaka Nagar centroid). `lm_x, lm_y` = `(-3155.2, 823.8)` (Ganesh Temple).

### Stage 4: ML Spatial Interpolation
- **Goal**: Predict the true location for *unvisited* addresses.
- **Input**: Baseline coordinates, Parsed features, Spatial Anchors, and Text Embeddings (TF-IDF + SVD).
- **Output**: Final predicted coordinates (`final_x`, `final_y`).
- **How it works**: A `HistGradientBoostingRegressor` is trained on the visited addresses. It learns the relationship between the text/anchors and the baseline's error (`delta_x`, `delta_y`). It then applies these corrections to unvisited addresses.
- **Example**:
  - *Input*: Unvisited address "Behind Ganesh Temple". Baseline says `(0, 0)`.
  - *Processing*: The model sees the "Ganesh Temple" anchor at `(-3155, 823)` and historical data showing baseline is usually off by -3000x for this area.
  - *Output*: `final_x, final_y` = `(-3150, 830)`.

### Stage 5: Confidence & Directions Generation
- **Goal**: Calculate the margin of error and generate human-readable instructions.
- **Input**: `final_x`, `final_y`, historical variance, and `landmarks_poi.csv`.
- **Output**: `confidence_radius`, `directions`, `tier`.
- **Example**:
  - *Input*: Location `(-3150, 830)`.
  - *Output*: 
    - `directions`: `"15m SW of Ganesh Temple"`
    - `confidence_radius`: `20m`
    - `tier`: `A` (Highly confident).
