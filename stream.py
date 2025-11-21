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

    # FIX: Convert latitude and longitude to numeric
    for coord in ["LATITUDE", "LONGITUDE"]:
        if coord in df.columns:
            df[coord] = pd.to_numeric(df[coord], errors="coerce")
        else:
            df[coord] = np.nan

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

# Get unique vehicle types safely
try:
    vehicle_options = sorted({vt for sub in df.get("VEHICLE_TYPES_LIST", [[]]) for vt in sub if pd.notna(vt) and str(vt).strip()})
except:
    vehicle_options = []
vehicles = st.sidebar.multiselect("Vehicle Type", options=vehicle_options)

# Get unique factors safely
try:
    factor_options = sorted({f for sub in df.get("FACTORS_LIST", [[]]) for f in sub if pd.notna(f) and str(f).strip()})
except:
    factor_options = []
factors = st.sidebar.multiselect("Contributing Factor", options=factor_options)

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
            dff["BOROUGH"].str.lower().str.contains(search_lower, na=False) |
            dff["PERSON_TYPE"].str.lower().str.contains(search_lower, na=False) |
            dff["PERSON_SEX"].str.lower().str.contains(search_lower, na=False) |
            dff["PERSON_INJURY"].str.lower().str.contains(search_lower, na=False)
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
    
    # FIX: Ensure numeric coordinates and handle missing values
    df_map = filtered_df.dropna(subset=["LATITUDE", "LONGITUDE"]).copy()
    df_map = df_map[
        (df_map["LATITUDE"].notna()) & 
        (df_map["LONGITUDE"].notna()) &
        (df_map["LATITUDE"] != 0) & 
        (df_map["LONGITUDE"] != 0)
    ]
    
    if not df_map.empty:
        # Ensure numeric types
        df_map["LATITUDE"] = pd.to_numeric(df_map["LATITUDE"], errors="coerce")
        df_map["LONGITUDE"] = pd.to_numeric(df_map["LONGITUDE"], errors="coerce")
        df_map = df_map.dropna(subset=["LATITUDE", "LONGITUDE"])
        
        if not df_map.empty:
            try:
                fig_map = px.scatter_mapbox(
                    df_map, 
                    lat="LATITUDE", 
                    lon="LONGITUDE", 
                    color="BOROUGH",
                    color_discrete_map=BOROUGH_COLORS,
                    hover_name="BOROUGH",
                    hover_data={
                        "TOTAL_INJURED": True, 
                        "TOTAL_KILLED": True,
                        "LATITUDE": False,
                        "LONGITUDE": False
                    },
                    zoom=9, 
                    height=500
                )
                fig_map.update_layout(
                    mapbox_style="open-street-map", 
                    margin=dict(t=0, b=0, l=0, r=0)
                )
                st.plotly_chart(fig_map, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating map: {str(e)}")
                st.info("Try adjusting your filters to get more location data")
        else:
            st.info("No valid location data available for current filters")
    else:
        st.info("No location data available for current filters")

    # Crash trends
    st.subheader("📈 Crash Trends Over Time")
    year_trend = filtered_df.groupby("YEAR").size().reset_index(name="Crashes")
    if not year_trend.empty:
        fig_trend = px.line(year_trend, x="YEAR", y="Crashes", markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No data available for trend analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏙️ Crashes by Borough")
        borough_counts = filtered_df.groupby("BOROUGH").size().reset_index(name="Count")
        if not borough_counts.empty:
            fig_borough = px.bar(borough_counts, x="BOROUGH", y="Count", 
                               color="BOROUGH", color_discrete_map=BOROUGH_COLORS)
            st.plotly_chart(fig_borough, use_container_width=True)
        else:
            st.info("No borough data available")
    
    with col2:
        st.subheader("💥 Injuries by Borough")
        borough_injuries = filtered_df.groupby("BOROUGH")["TOTAL_INJURED"].sum().reset_index()
        if not borough_injuries.empty:
            fig_injuries = px.bar(borough_injuries, x="BOROUGH", y="TOTAL_INJURED", 
                                color="BOROUGH", color_discrete_map=BOROUGH_COLORS)
            st.plotly_chart(fig_injuries, use_container_width=True)
        else:
            st.info("No injury data available by borough")

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 Contributing Factors")
        # Simplified factor analysis
        try:
            factor_data = []
            for factors_list in filtered_df.get("FACTORS_LIST", [[]]):
                if isinstance(factors_list, list):
                    factor_data.extend([f for f in factors_list if pd.notna(f) and str(f).strip()])
            
            if factor_data:
                factor_counts = pd.Series(factor_data).value_counts().head(10).reset_index()
                factor_counts.columns = ["Factor", "Count"]
                fig_factors = px.bar(factor_counts, x="Count", y="Factor", orientation='h')
                st.plotly_chart(fig_factors, use_container_width=True)
            else:
                st.info("No factor data available")
        except Exception as e:
            st.info("Could not generate factor analysis")
    
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
    try:
        vehicle_trend = filtered_df.groupby(["YEAR", "VEHICLE TYPE CODE 1"]).size().reset_index(name="Count")
        top_vehicles = vehicle_trend["VEHICLE TYPE CODE 1"].value_counts().head(5).index
        vehicle_trend = vehicle_trend[vehicle_trend["VEHICLE TYPE CODE 1"].isin(top_vehicles)]
        
        if not vehicle_trend.empty:
            fig_vehicle_trend = px.line(vehicle_trend, x="YEAR", y="Count", color="VEHICLE TYPE CODE 1")
            st.plotly_chart(fig_vehicle_trend, use_container_width=True)
        else:
            st.info("No vehicle trend data available")
    except:
        st.info("Could not generate vehicle trends")

# Continue with other tabs (shortened for brevity)
with tab3:
    st.subheader("👥 People & Injuries Analysis")
    st.info("People & Injuries tab content would go here...")

with tab4:
    st.subheader("📈 Demographics Analysis")
    st.info("Demographics tab content would go here...")

with tab5:
    st.subheader("🔬 Advanced Analytics")
    st.info("Advanced Analytics tab content would go here...")

# Footer
st.markdown("---")
st.markdown("**NYC Crash Analysis Dashboard | Built with Streamlit & Plotly**")

# Debug information (optional)
with st.sidebar:
    if st.checkbox("Show debug info"):
        st.write("Data shape:", filtered_df.shape)
        st.write("Columns:", filtered_df.columns.tolist())
        if "LATITUDE" in filtered_df.columns:
            st.write("Latitude sample:", filtered_df["LATITUDE"].head(3).tolist())
            st.write("Latitude dtype:", filtered_df["LATITUDE"].dtype)
