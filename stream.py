# nyc_crash_dashboard_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.cluster import KMeans
import ast
import re

# Set page config
st.set_page_config(
    page_title="NYC Crash Analysis Dashboard",
    page_icon="💥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    .metric-card {
        background-color: #FFE6E6;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #FF8DA1;
    }
</style>
""", unsafe_allow_html=True)

# Load and process data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/lusgad/nyc-crash-data/main/data_part_aa.gz"
    df = pd.read_csv(url, compression='gzip', dtype=str, low_memory=False)
    st.success(f"✅ Loaded {len(df)} rows")
    
    # Your existing data processing code
    borough_mapping = {
        'MANHATTAN': 'Manhattan',
        'BROOKLYN': 'Brooklyn',
        'QUEENS': 'Queens',
        'BRONX': 'Bronx',
        'STATEN ISLAND': 'Staten Island'
    }

    if "BOROUGH" in df.columns:
        df["BOROUGH"] = df["BOROUGH"].str.title().replace(borough_mapping)
        df["BOROUGH"] = df["BOROUGH"].fillna("Unknown")
    else:
        df["BOROUGH"] = "Unknown"

    # Convert crash datetime
    df["CRASH_DATETIME"] = pd.to_datetime(df["CRASH_DATETIME"], errors="coerce")
    df["YEAR"] = df["CRASH_DATETIME"].dt.year
    df["MONTH"] = df["CRASH_DATETIME"].dt.month
    df["HOUR"] = df["CRASH_DATETIME"].dt.hour
    df["DAY_OF_WEEK"] = df["CRASH_DATETIME"].dt.day_name()

    # Cast numeric columns
    num_cols = [
        "NUMBER OF PERSONS INJURED", "NUMBER OF PERSONS KILLED",
        "NUMBER OF PEDESTRIANS INJURED", "NUMBER OF PEDESTRIANS KILLED",
        "NUMBER OF CYCLIST INJURED", "NUMBER OF CYCLIST KILLED",
        "NUMBER OF MOTORIST INJURED", "NUMBER OF MOTORIST KILLED"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        else:
            df[c] = 0

    # Aggregated columns
    df["TOTAL_INJURED"] = df[["NUMBER OF PERSONS INJURED",
                             "NUMBER OF PEDESTRIANS INJURED",
                             "NUMBER OF CYCLIST INJURED",
                             "NUMBER OF MOTORIST INJURED"]].sum(axis=1)
    df["TOTAL_KILLED"] = df[["NUMBER OF PERSONS KILLED",
                            "NUMBER OF PEDESTRIANS KILLED",
                            "NUMBER OF CYCLIST KILLED",
                            "NUMBER OF MOTORIST KILLED"]].sum(axis=1)
    df["SEVERITY_SCORE"] = (df["TOTAL_INJURED"] * 1 + df["TOTAL_KILLED"] * 5)

    # Handle missing columns
    for col in ["PERSON_AGE", "PERSON_SEX", "BODILY_INJURY", "SAFETY_EQUIPMENT", 
                "EMOTIONAL_STATUS", "EJECTION", "ZIP CODE", "PERSON_INJURY"]:
        if col not in df.columns:
            if col == "PERSON_AGE":
                df[col] = 0
            else:
                df[col] = "Unknown"

    for col in ["COMPLAINT", "VEHICLE TYPE CODE 1", "CONTRIBUTING FACTOR VEHICLE 1"]:
        if col not in df.columns:
            df[col] = "Unknown"

    return df

# Load data
df = load_data()

# Sidebar filters
st.sidebar.header("📊 Control Panel")

# Year range slider
min_year = int(df["YEAR"].min()) if not df["YEAR"].isna().all() else 2010
max_year = int(df["YEAR"].max()) if not df["YEAR"].isna().all() else pd.Timestamp.now().year

year_range = st.sidebar.slider(
    "Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# Other filters
boroughs = st.sidebar.multiselect("Borough", options=sorted(df["BOROUGH"].unique()))
vehicles = st.sidebar.multiselect("Vehicle Type", options=sorted({vt for sub in df.get("VEHICLE_TYPES_LIST", [[]]) for vt in sub}))
factors = st.sidebar.multiselect("Contributing Factor", options=sorted({f for sub in df.get("FACTORS_LIST", [[]]) for f in sub}))
person_type = st.sidebar.multiselect("Person Type", options=sorted(df["PERSON_TYPE"].unique()))
injuries = st.sidebar.multiselect("Injury Type", options=sorted(df["PERSON_INJURY"].unique()))

# Search
search_text = st.sidebar.text_input("🔍 Advanced Search", placeholder="Try: 'queens bicycle pedestrian'...")

# Clear filters
if st.sidebar.button("🗑️ Clear All Filters"):
    st.rerun()

# Apply filters
def filter_data(df, year_range, boroughs, vehicles, factors, injuries, person_type, search_text):
    dff = df.copy()
    
    # Year filter
    dff = dff[(dff["YEAR"] >= year_range[0]) & (dff["YEAR"] <= year_range[1])]
    
    # Borough filter
    if boroughs:
        dff = dff[dff["BOROUGH"].isin(boroughs)]
    
    # Injury filter
    if injuries:
        dff = dff[dff["PERSON_INJURY"].isin(injuries)]
    
    # Person type filter
    if person_type:
        dff = dff[dff["PERSON_TYPE"].isin(person_type)]
    
    # Simple search implementation
    if search_text:
        search_lower = search_text.lower()
        mask = (
            dff["BOROUGH"].str.lower().str.contains(search_lower) |
            dff["PERSON_TYPE"].str.lower().str.contains(search_lower) |
            dff["PERSON_SEX"].str.lower().str.contains(search_lower) |
            dff["PERSON_INJURY"].str.lower().str.contains(search_lower)
        )
        dff = dff[mask]
    
    return dff

# Apply filtering
filtered_df = filter_data(df, year_range, boroughs, vehicles, factors, injuries, person_type, search_text)

# Main dashboard
st.markdown('<h1 class="main-header">💥 NYC Crash Analysis Dashboard</h1>', unsafe_allow_html=True)

# Summary metrics
total_crashes = len(filtered_df)
total_injuries = filtered_df["TOTAL_INJURED"].sum()
total_killed = filtered_df["TOTAL_KILLED"].sum()
avg_injuries = total_injuries / total_crashes if total_crashes > 0 else 0

st.markdown(f"**📊 Currently showing: {total_crashes:,} crashes | {total_injuries:,} injured | {total_killed:,} fatalities**")

# Metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🏎️ Total Crashes", f"{total_crashes:,}")
with col2:
    st.metric("💥 Total Injuries", f"{total_injuries:,}")
with col3:
    st.metric("💀 Total Fatalities", f"{total_killed:,}")
with col4:
    st.metric("📈 Avg Injuries/Crash", f"{avg_injuries:.2f}")

# Define consistent colors
BOROUGH_COLORS = {
    'Manhattan': '#2ECC71', 'Brooklyn': '#E74C3C', 'Queens': '#3498DB',
    'Bronx': '#F39C12', 'Staten Island': '#9B59B6', 'Unknown': '#95A5A6'
}

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Crash Geography", 
    "🏎️ Vehicles & Factors", 
    "👥 People & Injuries", 
    "📈 Demographics",
    "🔬 Advanced Analytics"
])

with tab1:
    st.subheader("📍 Crash Locations Map")
    df_map = filtered_df.dropna(subset=["LATITUDE", "LONGITUDE"])
    if not df_map.empty:
        fig_map = px.scatter_mapbox(
            df_map, 
            lat="LATITUDE", 
            lon="LONGITUDE", 
            color="BOROUGH",
            color_discrete_map=BOROUGH_COLORS,
            hover_name="BOROUGH",
            hover_data={"TOTAL_INJURED": True, "TOTAL_KILLED": True},
            zoom=9, 
            height=500
        )
        fig_map.update_layout(mapbox_style="open-street-map", margin=dict(t=0))
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No location data available for current filters")

    # Crash trends
    st.subheader("📈 Crash Trends Over Time")
    year_trend = filtered_df.groupby("YEAR").size().reset_index(name="Crashes")
    if not year_trend.empty:
        fig_trend = px.line(year_trend, x="YEAR", y="Crashes", markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏙️ Crashes by Borough")
        borough_counts = filtered_df.groupby("BOROUGH").size().reset_index(name="Count")
        fig_borough = px.bar(borough_counts, x="BOROUGH", y="Count", color="BOROUGH", color_discrete_map=BOROUGH_COLORS)
        st.plotly_chart(fig_borough, use_container_width=True)
    
    with col2:
        st.subheader("💥 Injuries by Borough")
        borough_injuries = filtered_df.groupby("BOROUGH")["TOTAL_INJURED"].sum().reset_index()
        fig_injuries = px.bar(borough_injuries, x="BOROUGH", y="TOTAL_INJURED", color="BOROUGH", color_discrete_map=BOROUGH_COLORS)
        st.plotly_chart(fig_injuries, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 Contributing Factors")
        # Simplified factor analysis
        factor_data = []
        for factors_list in filtered_df.get("FACTORS_LIST", [[]]):
            if isinstance(factors_list, list):
                factor_data.extend(factors_list)
        
        if factor_data:
            factor_counts = pd.Series(factor_data).value_counts().head(10).reset_index()
            factor_counts.columns = ["Factor", "Count"]
            fig_factors = px.bar(factor_counts, x="Count", y="Factor", orientation='h')
            st.plotly_chart(fig_factors, use_container_width=True)
        else:
            st.info("No factor data available")
    
    with col2:
        st.subheader("🔥 Vehicle vs Factor Heatmap")
        # Simplified heatmap
        try:
            top_vehicles = filtered_df["VEHICLE TYPE CODE 1"].value_counts().head(5).index
            top_factors = filtered_df["CONTRIBUTING FACTOR VEHICLE 1"].value_counts().head(5).index
            
            heatmap_data = filtered_df[
                filtered_df["VEHICLE TYPE CODE 1"].isin(top_vehicles) & 
                filtered_df["CONTRIBUTING FACTOR VEHICLE 1"].isin(top_factors)
            ]
            
            if not heatmap_data.empty:
                pivot_table = pd.crosstab(
                    heatmap_data["VEHICLE TYPE CODE 1"], 
                    heatmap_data["CONTRIBUTING FACTOR VEHICLE 1"]
                )
                fig_heatmap = px.imshow(pivot_table, aspect="auto")
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("Insufficient data for heatmap")
        except:
            st.info("Could not generate heatmap with current data")

    st.subheader("🏎️ Vehicle Type Trends")
    vehicle_trend = filtered_df.groupby(["YEAR", "VEHICLE TYPE CODE 1"]).size().reset_index(name="Count")
    top_vehicles = vehicle_trend["VEHICLE TYPE CODE 1"].value_counts().head(5).index
    vehicle_trend = vehicle_trend[vehicle_trend["VEHICLE TYPE CODE 1"].isin(top_vehicles)]
    
    if not vehicle_trend.empty:
        fig_vehicle_trend = px.line(vehicle_trend, x="YEAR", y="Count", color="VEHICLE TYPE CODE 1")
        st.plotly_chart(fig_vehicle_trend, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛡️ Safety Equipment")
        safety_counts = filtered_df["SAFETY_EQUIPMENT"].value_counts().head(5).reset_index()
        safety_counts.columns = ["Equipment", "Count"]
        fig_safety = px.pie(safety_counts, values="Count", names="Equipment")
        st.plotly_chart(fig_safety, use_container_width=True)
    
    with col2:
        st.subheader("🚑 Injury Types")
        injury_counts = filtered_df["BODILY_INJURY"].value_counts().head(10).reset_index()
        injury_counts.columns = ["Injury", "Count"]
        fig_injury = px.bar(injury_counts, x="Count", y="Injury", orientation='h')
        st.plotly_chart(fig_injury, use_container_width=True)

    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.subheader("🎭 Emotional State")
        emotional_counts = filtered_df["EMOTIONAL_STATUS"].value_counts().head(5).reset_index()
        emotional_counts.columns = ["State", "Count"]
        fig_emotional = px.bar(emotional_counts, x="State", y="Count")
        st.plotly_chart(fig_emotional, use_container_width=True)
    
    with col4:
        st.subheader("🚪 Ejection Status")
        ejection_counts = filtered_df["EJECTION"].value_counts().reset_index()
        ejection_counts.columns = ["Status", "Count"]
        fig_ejection = px.pie(ejection_counts, values="Count", names="Status")
        st.plotly_chart(fig_ejection, use_container_width=True)
    
    with col5:
        st.subheader("💺 Position in Vehicle")
        position_counts = filtered_df["POSITION_IN_VEHICLE_CLEAN"].value_counts().head(5).reset_index()
        position_counts.columns = ["Position", "Count"]
        fig_position = px.bar(position_counts, x="Position", y="Count")
        st.plotly_chart(fig_position, use_container_width=True)

    col6, col7 = st.columns(2)
    
    with col6:
        st.subheader("👥 Person Types Over Time")
        person_time = filtered_df.groupby(["YEAR", "PERSON_TYPE"]).size().reset_index(name="Count")
        fig_person_time = px.area(person_time, x="YEAR", y="Count", color="PERSON_TYPE")
        st.plotly_chart(fig_person_time, use_container_width=True)
    
    with col7:
        st.subheader("📋 Top Complaints")
        complaint_counts = filtered_df["COMPLAINT"].value_counts().head(10).reset_index()
        complaint_counts.columns = ["Complaint", "Count"]
        fig_complaint = px.bar(complaint_counts, x="Count", y="Complaint", orientation='h')
        st.plotly_chart(fig_complaint, use_container_width=True)

with tab4:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Age Distribution")
        filtered_df["PERSON_AGE"] = pd.to_numeric(filtered_df["PERSON_AGE"], errors='coerce')
        age_data = filtered_df["PERSON_AGE"].dropna()
        if not age_data.empty:
            fig_age = px.histogram(age_data, nbins=30, title="Age Distribution")
            st.plotly_chart(fig_age, use_container_width=True)
        else:
            st.info("No age data available")
    
    with col2:
        st.subheader("🚻 Gender Distribution")
        gender_counts = filtered_df["PERSON_SEX"].value_counts().reset_index()
        gender_counts.columns = ["Gender", "Count"]
        fig_gender = px.pie(gender_counts, values="Count", names="Gender")
        st.plotly_chart(fig_gender, use_container_width=True)

with tab5:
    st.subheader("🔬 Advanced Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Risk Correlation Matrix")
        try:
            numeric_cols = ['PERSON_AGE', 'TOTAL_INJURED', 'TOTAL_KILLED', 'SEVERITY_SCORE', 'HOUR']
            available_numeric = [col for col in numeric_cols if col in filtered_df.columns]
            
            if len(available_numeric) > 1:
                corr_matrix = filtered_df[available_numeric].corr()
                fig_corr = ff.create_annotated_heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns.tolist(),
                    y=corr_matrix.columns.tolist(),
                    annotation_text=corr_matrix.round(2).values,
                    colorscale='Blues'
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Not enough numeric data for correlation analysis")
        except:
            st.info("Could not generate correlation matrix")
    
    with col2:
        st.subheader("🕒 Temporal Risk Patterns")
        try:
            temporal_data = filtered_df.groupby(['DAY_OF_WEEK', 'HOUR']).size().reset_index(name='Count')
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            temporal_data['DAY_OF_WEEK'] = pd.Categorical(temporal_data['DAY_OF_WEEK'], categories=day_order, ordered=True)
            temporal_data = temporal_data.sort_values(['DAY_OF_WEEK', 'HOUR'])

            fig_temporal = px.density_heatmap(temporal_data, x='HOUR', y='DAY_OF_WEEK', z='Count',
                                           color_continuous_scale='viridis')
            st.plotly_chart(fig_temporal, use_container_width=True)
        except:
            st.info("Could not generate temporal patterns")

    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🎯 Severity Prediction Factors")
        try:
            severity_factors = filtered_df.groupby('BOROUGH')['SEVERITY_SCORE'].mean().reset_index()
            fig_severity = px.bar(severity_factors, x='BOROUGH', y='SEVERITY_SCORE', 
                                color='BOROUGH', color_discrete_map=BOROUGH_COLORS)
            st.plotly_chart(fig_severity, use_container_width=True)
        except:
            st.info("Could not generate severity analysis")
    
    with col4:
        st.subheader("🔥 Crash Hotspot Clustering")
        df_coords = filtered_df.dropna(subset=["LATITUDE", "LONGITUDE"])
        if len(df_coords) > 10:
            try:
                coords = df_coords[["LATITUDE", "LONGITUDE"]].values
                kmeans = KMeans(n_clusters=min(10, len(df_coords)), random_state=42)
                df_coords["CLUSTER"] = kmeans.fit_predict(coords)
                cluster_sizes = df_coords.groupby("CLUSTER").size()
                df_coords["CLUSTER_SIZE"] = df_coords["CLUSTER"].map(cluster_sizes)

                fig_hotspot = px.scatter_mapbox(df_coords, lat="LATITUDE", lon="LONGITUDE",
                                             color="CLUSTER_SIZE", size="CLUSTER_SIZE",
                                             zoom=9, height=400)
                fig_hotspot.update_layout(mapbox_style="open-street-map")
                st.plotly_chart(fig_hotspot, use_container_width=True)
            except:
                st.info("Could not generate hotspot clustering")
        else:
            st.info("Not enough location data for clustering")

# Footer
st.markdown("---")
st.markdown("**NYC Crash Analysis Dashboard | Built with Streamlit & Plotly**")
