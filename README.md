# CreditNirvana: Geocoding Engine

This repository contains the solution for an **Address Geocoder that Learns from Field Visits**. The engine computes a probabilistic location for each address by fusing commercial pin data, address text, and field visit GPS traces into a single Bayesian Grid Posterior model.

## Professional Repository Structure

```
.
├── backend/                  # FastAPI Application for the Dashboard
├── frontend/                 # Next.js React Dashboard
├── data/
│   ├── raw/                  # Original datasets
│   └── processed/            # Processed geocoded CSVs and output features
├── docs/                     # Reports, charts, and diagrams
├── notebooks/                # Exploratory Data Analysis
└── src/                      # Source code
    ├── data_processing/      # Scripts for data extraction
    ├── eda/                  # EDA and trends generators
    ├── models/               # Geocoder logic and modeling pipelines
    └── utils/                # Utility scripts
```

## Dashboard Application

A professional dashboard is provided to search for addresses, review their tiers, geocodes, confidence radius, and generated directions.

### Running the API Backend

Navigate to the `backend` directory, install requirements, and run the server:
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
The API will run at `http://localhost:8000`.

### Running the Frontend Dashboard

Navigate to the `frontend` directory, install dependencies, and run the development server:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000` in your browser.

## Pipeline Architecture

Below is the workflow of the professional geocoding pipeline implemented in `src/models/geocoding_engine.py`:

```mermaid
flowchart TD
    %% Define styles
    classDef input fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px,color:#000;
    classDef process fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px,color:#000;
    classDef model fill:#fff3e0,stroke:#ff9800,stroke-width:2px,color:#000;
    classDef logic fill:#e8f5e9,stroke:#4caf50,stroke-width:2px,color:#000;
    classDef output fill:#ffebee,stroke:#f44336,stroke-width:2px,color:#000;
    classDef drop fill:#eceff1,stroke:#607d8b,stroke-width:2px,stroke-dasharray: 5 5,color:#000;

    %% Data Inputs
    A1[(addresses.csv)]:::input
    A2[(field_visits.csv)]:::input
    A3[(baseline_geocodes.csv)]:::input
    A4[(landmarks_poi.csv)]:::input

    %% Processing Nodes
    B1[Address Parser<br>Extract Pincode/House No]:::process
    B2[Fraud & Outlier Detector<br>Filter Hash Spoofs & Distance Outliers]:::process
    
    %% Intermediate states
    C1{Valid Field Visits?}:::logic
    C2[Extract Ground Truth<br>Clean Visit Check-ins]:::process
    
    %% Modeling (Bayesian Grid)
    D1[Bayesian Grid Posterior Model<br>Generate 330x330 Grid]:::model
    
    %% Likelihood Components
    D2[Commercial Pin Likelihood<br>Gaussian Prior]:::logic
    D3[Field Visit Likelihood<br>Quality-Adjusted Gaussian]:::logic

    %% Inference
    D4[Log Posterior Accumulation<br>Weighted Geometric Median Approximation]:::model

    %% Feature Generation
    E1[Calculate Confidence Radius & Tiers]:::process
    E2[Generate POI Directions<br>Math.atan2 + Distance]:::process
    
    %% Output
    F1[(Geocoding_Engine_Output.csv)]:::output

    %% Edges
    A1 --> B1
    A2 --> B2
    A3 --> B2
    B2 --> C1
    
    C1 -- Yes --> C2
    C1 -- Spoofed/Invalid --> B2_Drop[Discard Fake GPS]:::drop
    
    B1 --> D1
    C2 --> D1
    A3 --> D1
    
    D1 --> D2
    D1 --> D3
    D2 --> D4
    D3 --> D4
    
    D4 --> E1
    A4 --> E2
    E1 --> E2
    E2 --> F1
```

## Output Features
The final pipeline fulfills all problem statement requirements by generating:
- **`x`, `y`**: The corrected location pin computed via the Bayesian Grid.
- **`confidence_radius`**: Confidence radius mathematically derived from check-in variance or global spread.
- **`tier`**: Visit priority classification (`A`: direct visit, `B`: visit with directions, `C`: verify first).
- **`directions`**: Computed using Cartesian distance and compass-angle (atan2) relative to nearest landmark POIs (e.g., `243m NE of Hanuman Temple`).
- **`address_text`**: The original full address for frontend display.
