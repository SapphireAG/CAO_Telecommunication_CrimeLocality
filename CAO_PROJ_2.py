import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from streamlit_autorefresh import st_autorefresh   # ✅ optional helper if you installed it


st.set_page_config(page_title="Crime Clustering - Real Time", layout="wide")
st.markdown("<h2 style='text-align:center'> Real-Time Crime Clustering </h2>", unsafe_allow_html=True)


data = pd.read_csv("/Users/amangolani/Downloads/CAO/crime-cast-forecasting-crime-categories/train.csv")


kmeans = MiniBatchKMeans(n_clusters=5, random_state=42)
categories = data['Crime_Category'].unique()
palette = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
category_colors = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}


st.sidebar.markdown("### Crime Category Colors")
for cat, color in category_colors.items():
    st.sidebar.markdown(f"<span style='color:{color}'>●</span> {cat}", unsafe_allow_html=True)


count = st_autorefresh(interval=2000, limit=100, key="refresh")  # every 2 sec refresh
batch_size = 10

# Keep count in session_state
if "idx" not in st.session_state:
    st.session_state.idx = 0
st.session_state.idx = min(st.session_state.idx + batch_size, len(data))

subset = data.iloc[:st.session_state.idx]


if len(subset) >= 5:
    X = subset[["Latitude", "Longitude"]].to_numpy()
    kmeans.partial_fit(X)
    labels = kmeans.predict(X)
    subset["Cluster"] = labels


    center_lat, center_lon = subset["Latitude"].mean(), subset["Longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9)
    for _, r in subset.iterrows():
        folium.CircleMarker(
            location=[r["Latitude"], r["Longitude"]],
            radius=4,
            color=category_colors.get(r["Crime_Category"], "gray"),
            fill=True,
            fill_opacity=0.6,
            popup=f"{r['Crime_Category']} (Cluster {r['Cluster']})"
        ).add_to(m)


    counts = subset["Crime_Category"].value_counts().reset_index()
    counts.columns = ["Category", "Count"]

    col1, col2 = st.columns([2, 1])
    with col1:
        st_folium(m, width=800, height=500)
    with col2:
        st.subheader("📊 Category Distribution")
        st.bar_chart(counts.set_index("Category"))
        st.metric("Total Crimes Processed", len(subset))
        st.metric("Active Clusters", len(np.unique(labels)))

else:
    st.info("Waiting for enough data points to start clustering...")
