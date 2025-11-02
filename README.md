# CAO_Telecommunication_CrimeLocality

# Real-Time Crime Clustering using Streamlit, Folium & MiniBatch K-Means

## Project Overview
This project simulates a real-time data processing pipeline for crime reports across U.S. localities.  
It demonstrates how OpenMP-style parallelism and pipelining concepts can be applied to a machine-learning and data-visualization workflow.

The system ingests crime location data, incrementally clusters it using `MiniBatchKMeans`, and visualizes evolving clusters on an interactive Folium map and a live-updating bar chart.  
Each pipeline stage corresponds to a CPU instruction pipeline stage — Fetch, Decode, Execute, Write-Back — making it an ideal CAO demonstration of concurrency, streaming, and state retention.

---

## Tech Stack
- Python (core language)  
- Streamlit — interactive real-time dashboard  
- Folium + streamlit-folium — geospatial visualization  
- scikit-learn (MiniBatchKMeans) — incremental clustering  
- Pandas, NumPy — data handling  
- Matplotlib / Plotly (optional) — supplementary graphs  

---

## Architecture and Pipelining Analogy

| CAO Concept | Data Pipeline Stage | Implementation |
|--------------|--------------------|----------------|
| Instruction Fetch | Crime data ingestion | CSV batches read sequentially |
| Instruction Decode | Feature extraction (Latitude–Longitude) | Pandas preprocessing |
| Execute | K-Means clustering | Incremental partial fit (`MiniBatchKMeans`) |
| Write-Back | Visualization and metrics | Folium map + bar chart update |

This models **task parallelism** (different modules running concurrently) and **pipeline parallelism** (continuous flow of data) — similar to OpenMP’s `sections` and `task` constructs.


---

## Dashboard Features
- Interactive map view showing current cluster distribution  
- Category color legend to differentiate crime types  
- Live bar chart displaying count per crime category  
- “Next Batch” control for smooth, stepwise updates without flicker  
- Incremental machine-learning computation to simulate streaming data ingestion  

---

## Data Source
Dataset: `train.csv`  
Each record includes:
- `Latitude` / `Longitude` — location coordinates  
- `Crime_Category` — type of reported crime  

---

## Key Learnings
- Mapping OpenMP parallel concepts to modern data pipelines  
- Designing a low-latency, stateful Streamlit visualization  
- Applying incremental machine learning for real-time data streams  
- Understanding concurrency and pipelining through visualization  

---

## To Run the App
```bash
streamlit run CAO_PROJ_2.py
