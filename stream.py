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

# Set page config with your exact colors and layout
st.set_page_config(
    page_title="NYC Crash Analysis Dashboard",
    page_icon="💥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Your exact CSS styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #cee8f0;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        color: #ffffff;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Cards */
    .metric-card {
        background-color: #FFE6E6;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #FF8DA1;
        margin: 10px 0;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1lcbmhc {
        background-color: #add8e6;
    }
    
    /* Tabs styling to match your custom tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #FF8DA1;
        color: white;
        border: 1px solid #FF8DA1;
        font-weight: bold;
        margin-right: 5px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #FF8DA1;
        border: 1px solid #FF8DA1;
    }
    
    /* Update button styling */
    .update-button {
        background-color: #FF8DA1 !important;
        color: white !important;
        border: none !important;
        font-weight: bold !important;
        padding: 10px 20px !important;
        border-radius: 5px !important;
    }
</style>
""", unsafe_allow_html=True)

# Load and process data (your exact data processing code)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/lusgad/nyc-crash-data/main/data_part_aa.gz"
    df = pd.read_csv(url, compression='gzip', dtype=str, low_memory=False)
    
    # Your exact data processing code
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

    # Cast numeric injury/killed counts to numeric (safe)
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

    # Helpful aggregated numeric columns
    df["TOTAL_INJURED"] = df[["NUMBER OF PERSONS INJURED",
                             "NUMBER OF PEDESTRIANS INJURED",
                             "NUMBER OF CYCLIST INJURED",
                             "NUMBER OF MOTORIST INJURED"]].sum(axis=1)
    df["TOTAL_KILLED"] = df[["NUMBER OF PERSONS KILLED",
                            "NUMBER OF PEDESTRIANS KILLED",
                            "NUMBER OF CYCLIST KILLED",
                            "NUMBER OF MOTORIST KILLED"]].sum(axis=1)

    # Create severity score for advanced analysis
    df["SEVERITY_SCORE"] = (df["TOTAL_INJURED"] * 1 + df["TOTAL_KILLED"] * 5)

    # FULL_ADDRESS fallback
    if "FULL ADDRESS" not in df.columns:
        df["FULL ADDRESS"] = df.get("ON STREET NAME", "").fillna("") + ", " + df.get("BOROUGH", "")

    # Latitude / Longitude as numeric
    for coord in ("LATITUDE", "LONGITUDE"):
        if coord in df.columns:
            df[coord] = pd.to_numeric(df[coord], errors="coerce")
        else:
            df[coord] = np.nan

    # Parse vehicle types
    def parse_vehicle_list(v):
        if pd.isna(v):
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        s = str(v).strip()
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            parts = [p.strip() for p in s.split(",") if p.strip()]
            return parts
        return []

    df["VEHICLE_TYPES_LIST"] = df.get("ALL_VEHICLE_TYPES", "").apply(parse_vehicle_list)

    # Parse contributing factors
    def parse_factor_list(v):
        if pd.isna(v):
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        s = str(v).strip()
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            parts = [p.strip() for p in s.split(",") if p.strip()]
            return parts
        return []

    if "ALL_CONTRIBUTING_FACTORS" in df.columns:
        df["FACTORS_LIST"] = df["ALL_CONTRIBUTING_FACTORS"].apply(parse_factor_list)
    elif "ALL_CONTRIBUTING_FACTORS_STR" in df.columns:
        df["FACTORS_LIST"] = df["ALL_CONTRIBUTING_FACTORS_STR"].apply(parse_factor_list)
    else:
        parts = []
        for i in range(1, 4):
            c = f"CONTRIBUTING FACTOR VEHICLE {i}"
            if c in df.columns:
                parts.append(df[c].fillna("").astype(str))
        if parts:
            df["FACTORS_LIST"] = (pd.Series([";".join(x) for x in zip(*parts)]) if parts else pd.Series([[]]*len(df))).apply(
                lambda s: parse_factor_list(s))
        else:
            df["FACTORS_LIST"] = [[] for _ in range(len(df))]

    # Ensure other person-related columns exist
    for col in ["PERSON_AGE", "PERSON_SEX", "BODILY_INJURY", "SAFETY_EQUIPMENT", "EMOTIONAL_STATUS", "UNIQUE_ID", "EJECTION", "ZIP CODE", "PERSON_INJURY"]:
        if col not in df.columns:
            if col == "UNIQUE_ID":
                df[col] = df.index + 1
            elif col == "PERSON_AGE":
                df[col] = pd.to_numeric(df.get(col, np.nan), errors='coerce').fillna(0).astype(int)
            elif col in ["EJECTION", "ZIP CODE", "PERSON_INJURY"]:
                df[col] = df.get(col, "Unknown").fillna("Unknown")
            else:
                df[col] = df.get(col, "Unknown").fillna("Unknown")

    # Ensure additional columns exist
    for col in ["COMPLAINT", "VEHICLE TYPE CODE 1", "CONTRIBUTING FACTOR VEHICLE 1"]:
        if col not in df.columns:
            df[col] = "Unknown"

    return df

# Load data
df = load_data()
st.success(f"✅ Loaded {len(df)} rows from dataset")

# Define consistent borough colors with proper capitalization
BOROUGH_COLORS = {
    'Manhattan': '#2ECC71',  # Green
    'Brooklyn': '#E74C3C',   # Red
    'Queens': '#3498DB',     # Blue
    'Bronx': '#F39C12',      # Orange
    'Staten Island': '#9B59B6', # Purple
    'Unknown': '#95A5A6'     # Gray
}

# Your exact search query function
def parse_search_query(q):
    q = (q or "").lower().strip()
    found = {}

    # Extract years and year ranges
    year_pattern = r'\b(20\d{2})\b'
    years_found = re.findall(year_pattern, q)
    if years_found:
        years = sorted([int(y) for y in years_found])
        if len(years) >= 2:
            found["year_range"] = [years[0], years[-1]]
        else:
            found["year"] = years[0]

    # Borough detection
    borough_keywords = {
        'manhattan': 'Manhattan',
        'brooklyn': 'Brooklyn',
        'queens': 'Queens',
        'bronx': 'Bronx',
        'staten': 'Staten Island',
        'staten island': 'Staten Island'
    }
    for keyword, borough in borough_keywords.items():
        if keyword in q:
            found["borough"] = [borough]
            break

    # Vehicle type detection
    vehicle_keywords = {
        'suv': 'SUV/Station Wagon',
        'station wagon': 'SUV/Station Wagon',
        'sedan': 'Sedan',
        'bicycle': 'Bicycle',
        'bike': 'Bicycle',
        'ambulance': 'Ambulance',
        'bus': 'Bus',
        'motorcycle': 'Motorcycle',
        'pickup': 'Pickup Truck',
        'pickup truck': 'Pickup Truck',
        'taxi': 'Taxi',
        'truck': 'Truck/Commercial',
        'commercial': 'Truck/Commercial',
        'van': 'Van',
        'pedicab': 'Pedicab'
    }
    vehicle_matches = []
    for keyword, vehicle_type in vehicle_keywords.items():
        if keyword in q:
            vehicle_matches.append(vehicle_type)
    if vehicle_matches:
        found["vehicle"] = vehicle_matches

    # Person type and demographics
    person_type_matches = []
    if 'pedestrian' in q:
        person_type_matches.append('Pedestrian')
    if 'cyclist' in q or 'bicyclist' in q:
        person_type_matches.append('Bicyclist')
    if 'motorist' in q:
        person_type_matches.append('Motorist')
    if 'driver' in q:
        person_type_matches.append('Driver')
    if 'occupant' in q:
        person_type_matches.append('Occupant')
    if person_type_matches:
        found["person_type"] = person_type_matches

    # Gender detection
    if 'female' in q or 'woman' in q or 'women' in q:
        found["gender"] = ['F']
    elif 'male' in q or 'man' in q or 'men' in q:
        found["gender"] = ['M']

    # Direct F/M detection
    if ' f ' in f" {q} " or q.endswith(' f') or q.startswith('f '):
        found["gender"] = ['F']
    elif ' m ' in f" {q} " or q.endswith(' m') or q.startswith('m '):
        found["gender"] = ['M']

    # Injury severity
    injury_matches = []
    if 'injured' in q or 'injury' in q:
        injury_matches.append('Injured')
    if 'killed' in q or 'fatal' in q or 'fatality' in q or 'death' in q or 'died' in q:
        injury_matches.append('Killed')
    if 'unspecified' in q:
        injury_matches.append('Unspecified')
    if injury_matches:
        found["injury"] = injury_matches

    return found

# Small helper to add jitter to lat/lon to separate overlapping points
def jitter_coords(series, scale=0.0006):
    return series + np.random.normal(loc=0, scale=scale, size=series.shape)

# Define pink plot template (FIXED: separate layout dict)
pink_template = {
    'paper_bgcolor': '#FFE6E6',
    'plot_bgcolor': '#FFE6E6',
    'font': {'color': '#2C3E50'},
    'xaxis': {'gridcolor': '#FFB6C1', 'linecolor': '#2C3E50'},
    'yaxis': {'gridcolor': '#FFB6C1', 'linecolor': '#2C3E50'}
}

# Vibrant color sequence for line charts
vibrant_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']

# ===== SESSION STATE FOR UPDATE BUTTON =====
if 'filters_applied' not in st.session_state:
    st.session_state.filters_applied = False

# ===== SIDEBAR FILTERS =====
with st.sidebar:
    st.markdown("""
    <div style='background-color: #FF8DA1; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
        <h4 style='color: white; margin: 0;'>📊 Control Panel</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Year Range Slider
    min_year = int(df["YEAR"].min()) if not df["YEAR"].isna().all() else 2010
    max_year = int(df["YEAR"].max()) if not df["YEAR"].isna().all() else pd.Timestamp.now().year
    
    year_range = st.slider(
        "Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        help="Select the year range for analysis"
    )
    
    # Filters Row 1
    col1, col2 = st.columns(2)
    with col1:
        boroughs = st.multiselect(
            "Borough",
            options=sorted(df["BOROUGH"].dropna().unique()),
            placeholder="All Boroughs"
        )
    with col2:
        # Get unique vehicle types
        all_vehicles = sorted({vt for sub in df["VEHICLE_TYPES_LIST"] for vt in sub if vt})
        vehicles = st.multiselect(
            "Vehicle Type",
            options=all_vehicles,
            placeholder="All Vehicle Types"
        )
    
    # Filters Row 2
    col3, col4 = st.columns(2)
    with col3:
        # Get unique factors
        all_factors = sorted({f for sub in df["FACTORS_LIST"] for f in sub if f})
        factors = st.multiselect(
            "Contributing Factor",
            options=all_factors,
            placeholder="All Factors"
        )
    with col4:
        person_type = st.multiselect(
            "Person Type",
            options=sorted(df["PERSON_TYPE"].dropna().unique()),
            placeholder="All Person Types"
        )
    
    # Injury Filter
    injuries = st.multiselect(
        "Injury Type",
        options=sorted(df["PERSON_INJURY"].dropna().unique()),
        placeholder="All Injury Types"
    )
    
    # Search Section
    st.markdown("**🔍 Advanced Search**")
    search_text = st.text_input(
        "Search",
        placeholder="Try: 'queens 2019 to 2022 bicycle female pedestrian'...",
        label_visibility="collapsed"
    )
    st.caption("Search by borough, year, vehicle type, gender, injury type")
    
    # Update Report Button
    if st.button("🔄 UPDATE DASHBOARD", use_container_width=True, type="primary"):
        st.session_state.filters_applied = True
        st.rerun()
    
    # Clear Filters Button
    if st.button("🗑️ Clear All Filters", use_container_width=True):
        st.session_state.filters_applied = False
        st.rerun()

# ===== APPLY FILTERS =====
def apply_filters(df, year_range, boroughs, vehicles, factors, injuries, person_type, search_text):
    dff = df.copy()
    
    # Apply year range
    dff = dff[(dff["YEAR"] >= year_range[0]) & (dff["YEAR"] <= year_range[1])]
    
    # Apply borough filter
    if boroughs:
        dff = dff[dff["BOROUGH"].isin(boroughs)]
    
    # Apply injury filter
    if injuries:
        dff = dff[dff["PERSON_INJURY"].fillna("").astype(str).isin([str(i) for i in injuries])]
    
    # Apply vehicle filter
    if vehicles:
        mask = dff["VEHICLE_TYPES_LIST"].apply(lambda lst: any(v in (lst if isinstance(lst, list) else []) for v in vehicles))
        dff = dff[mask]
    
    # Apply contributing factor filter
    if factors:
        clean_factors = [str(f).strip().strip("[]'\"") for f in factors]
        mask = dff["FACTORS_LIST"].apply(lambda lst: any(
            any(clean_f in str(fact).strip().strip("[]'\"") for clean_f in clean_factors)
            for fact in (lst if isinstance(lst, list) else [])
        ))
        dff = dff[mask]
    
    # Apply person type filter
    if person_type:
        dff = dff[dff["PERSON_TYPE"].isin(person_type)]
    
    # Apply search query
    if search_text:
        parsed = parse_search_query(search_text)
        
        # Year range from search
        if "year_range" in parsed:
            yr_range = parsed["year_range"]
            year_range = [max(year_range[0], yr_range[0]), min(year_range[1], yr_range[1])]
        elif "year" in parsed:
            yr = parsed["year"]
            year_range = [max(year_range[0], yr), min(year_range[1], yr)]
        
        # Borough filter from search
        if "borough" in parsed:
            if boroughs:
                boroughs = list(set(boroughs) & set(parsed["borough"]))
            else:
                boroughs = parsed["borough"]
        
        # Vehicle filter from search
        if "vehicle" in parsed:
            if vehicles:
                vehicles = list(set(vehicles) & set(parsed["vehicle"]))
            else:
                vehicles = parsed["vehicle"]
        
        # Person type filter from search
        if "person_type" in parsed:
            if person_type:
                person_type = list(set(person_type) & set(parsed["person_type"]))
            else:
                person_type = parsed["person_type"]
        
        # Injury filter from search
        if "injury" in parsed:
            if injuries:
                injuries = list(set(injuries) & set(parsed["injury"]))
            else:
                injuries = parsed["injury"]
        
        # Gender filter from search
        if "gender" in parsed:
            gender_filter = parsed["gender"]
            dff = dff[dff["PERSON_SEX"].isin(gender_filter)]
    
    return dff

# Apply filtering - only when update button is clicked or initial load
if not st.session_state.filters_applied:
    # Show initial data (no filters applied except year range)
    filtered_df = df.copy()
    filtered_df = filtered_df[(filtered_df["YEAR"] >= year_range[0]) & (filtered_df["YEAR"] <= year_range[1])]
else:
    # Apply all filters when update button is clicked
    filtered_df = apply_filters(df, year_range, boroughs, vehicles, factors, injuries, person_type, search_text)

# ===== MAIN DASHBOARD =====

# Header with Summary
st.markdown('<h1 class="main-header">💥 NYC Crash Analysis Dashboard</h1>', unsafe_allow_html=True)

# Summary Text
total_crashes = len(filtered_df)
total_injuries = filtered_df["TOTAL_INJURED"].sum()
total_killed = filtered_df["TOTAL_KILLED"].sum()
summary_text = f"📊 Currently showing: {total_crashes:,} crashes | {total_injuries:,} injured | {total_killed:,} fatalities"

st.markdown(f"""
<div style='background-color: #FF8DA1; color: white; padding: 15px; border-radius: 10px; text-align: center; font-size: 18px; font-weight: bold; margin-bottom: 20px;'>
    {summary_text}
</div>
""", unsafe_allow_html=True)

# Live Statistics (your exact design)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div style='background-color: #FFE6E6; padding: 20px; border-radius: 10px; border: 2px solid #FF8DA1; text-align: center;'>
        <h4 style='color: #FF8DA1; margin: 0;'>🏎️ Total Crashes</h4>
        <h2 style='color: #FF8DA1; margin: 10px 0;'>{:,}</h2>
    </div>
    """.format(total_crashes), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='background-color: #FFE6E6; padding: 20px; border-radius: 10px; border: 2px solid #FF8DA1; text-align: center;'>
        <h4 style='color: #FF8DA1; margin: 0;'>💥 Total Injuries</h4>
        <h2 style='color: #FF8DA1; margin: 10px 0;'>{:,}</h2>
    </div>
    """.format(total_injuries), unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='background-color: #FFE6E6; padding: 20px; border-radius: 10px; border: 2px solid #FF8DA1; text-align: center;'>
        <h4 style='color: #FF8DA1; margin: 0;'>💀 Total Fatalities</h4>
        <h2 style='color: #FF8DA1; margin: 10px 0;'>{:,}</h2>
    </div>
    """.format(total_killed), unsafe_allow_html=True)

with col4:
    avg_injuries = total_injuries / total_crashes if total_crashes > 0 else 0
    st.markdown("""
    <div style='background-color: #FFE6E6; padding: 20px; border-radius: 10px; border: 2px solid #FF8DA1; text-align: center;'>
        <h4 style='color: #FF8DA1; margin: 0;'>📈 Avg Injuries/Crash</h4>
        <h2 style='color: #FF8DA1; margin: 10px 0;'>{:.2f}</h2>
    </div>
    """.format(avg_injuries), unsafe_allow_html=True)

# ===== TABBED INTERFACE =====
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Crash Geography", 
    "🏎️ Vehicles & Factors", 
    "👥 People & Injuries", 
    "📈 Demographics",
    "🔬 Advanced Analytics"
])

# Helper function to apply pink template
def apply_pink_template(fig):
    fig.update_layout(
        paper_bgcolor=pink_template['paper_bgcolor'],
        plot_bgcolor=pink_template['plot_bgcolor'],
        font=pink_template['font']
    )
    return fig

# ===== TAB 1: CRASH GEOGRAPHY =====
with tab1:
    # Crash Map - Full Width
    st.subheader("📍 Crash Locations Map")
    
    df_map = filtered_df.dropna(subset=["LATITUDE", "LONGITUDE"]).copy()
    if not df_map.empty:
        # Apply jitter like your original code
        df_map["_LAT_JIT"] = jitter_coords(df_map["LATITUDE"].fillna(0).astype(float), scale=0.0005)
        df_map["_LON_JIT"] = jitter_coords(df_map["LONGITUDE"].fillna(0).astype(float), scale=0.0005)
        
        fig_map = px.scatter_mapbox(
            df_map, 
            lat="_LAT_JIT", 
            lon="_LON_JIT", 
            color="BOROUGH",
            hover_name="FULL ADDRESS",
            hover_data={
                "FULL ADDRESS": True,
                "TOTAL_INJURED": True,
                "TOTAL_KILLED": True,
                "CRASH_DATETIME": True,
                "_LAT_JIT": False, 
                "_LON_JIT": False
            },
            zoom=9, 
            height=500,
            color_discrete_map=BOROUGH_COLORS
        )
        fig_map.update_traces(marker=dict(size=8, opacity=0.7))
        
        # FIXED: Update layout without duplicate parameters
        fig_map.update_layout(
            mapbox_style="open-street-map",
            margin=dict(t=0)
        )
        fig_map = apply_pink_template(fig_map)
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No location data to display for current filters")
    
    # Crash Trends - Full Width
    st.subheader("📈 Crash Trends Over Time")
    year_group = filtered_df.groupby(["YEAR", "BOROUGH"]).size().reset_index(name="Crashes")
    if not year_group.empty:
        fig_year = px.line(
            year_group, 
            x="YEAR", 
            y="Crashes", 
            color="BOROUGH", 
            markers=True,
            color_discrete_map=BOROUGH_COLORS
        )
        fig_year = apply_pink_template(fig_year)
        st.plotly_chart(fig_year, use_container_width=True)
    else:
        st.info("No data available for crash trends")
    
    # Borough Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏙️ Crashes by Borough")
        borough_df = filtered_df.groupby("BOROUGH").size().reset_index(name="Count").sort_values("Count", ascending=False)
        if not borough_df.empty:
            fig_borough = px.bar(
                borough_df, 
                x="BOROUGH", 
                y="Count",
                color="BOROUGH",
                color_discrete_map=BOROUGH_COLORS
            )
            fig_borough = apply_pink_template(fig_borough)
            fig_borough.update_layout(showlegend=False)
            st.plotly_chart(fig_borough, use_container_width=True)
    
    with col2:
        st.subheader("💥 Injuries by Borough")
        injuries_by_borough = filtered_df.groupby("BOROUGH")["TOTAL_INJURED"].sum().reset_index().sort_values("TOTAL_INJURED", ascending=False)
        if not injuries_by_borough.empty:
            fig_inj_borough = px.bar(
                injuries_by_borough, 
                x="BOROUGH", 
                y="TOTAL_INJURED",
                labels={"TOTAL_INJURED": "Total Injured", "BOROUGH": "Borough"},
                text="TOTAL_INJURED",
                color="BOROUGH",
                color_discrete_map=BOROUGH_COLORS
            )
            fig_inj_borough.update_traces(textposition="outside")
            fig_inj_borough = apply_pink_template(fig_inj_borough)
            fig_inj_borough.update_layout(
                margin=dict(t=40, b=20),
                showlegend=False
            )
            st.plotly_chart(fig_inj_borough, use_container_width=True)

# ===== TAB 2: VEHICLES & FACTORS =====
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 Contributing Factors")
        # Your exact factor analysis code
        factor_rows = []
        for _, row in filtered_df.iterrows():
            for f in row["FACTORS_LIST"]:
                factor_rows.append((f, row["UNIQUE_ID"] if "UNIQUE_ID" in row else 1))
        
        if factor_rows:
            factor_df = pd.DataFrame(factor_rows, columns=["Factor", "UID"])
            factor_counts_df = factor_df["Factor"].value_counts().head(15).reset_index()
            factor_counts_df.columns = ["Factor", "Count"]
            
            fig_factor = px.bar(
                factor_counts_df, 
                x="Count", 
                y="Factor", 
                orientation="h",
                labels={"Count": "Number of Crashes", "Factor": "Contributing Factor"},
                color="Count",
                color_continuous_scale="purples"
            )
            fig_factor = apply_pink_template(fig_factor)
            fig_factor.update_layout(
                margin=dict(t=40, b=20), 
                yaxis={'categoryorder':'total ascending'}
            )
            st.plotly_chart(fig_factor, use_container_width=True)
        else:
            st.info("No factor data available")
    
    with col2:
        st.subheader("🔥 Vehicle vs Factor Heatmap")
        # Your exact heatmap code
        top_factors = filtered_df["CONTRIBUTING FACTOR VEHICLE 1"].value_counts().nlargest(10).index
        fig_vehicle_factor = px.density_heatmap(
            filtered_df[filtered_df["CONTRIBUTING FACTOR VEHICLE 1"].isin(top_factors)],
            x="VEHICLE TYPE CODE 1",
            y="CONTRIBUTING FACTOR VEHICLE 1",
            labels={
                "VEHICLE TYPE CODE 1": "Vehicle Type", 
                "CONTRIBUTING FACTOR VEHICLE 1": "Contributing Factor"
            },
            color_continuous_scale="viridis"
        )
        fig_vehicle_factor = apply_pink_template(fig_vehicle_factor)
        st.plotly_chart(fig_vehicle_factor, use_container_width=True)
    
    st.subheader("🏎️ Vehicle Type Trends")
    # Your exact vehicle trend code
    trend_df = filtered_df.groupby(["YEAR", "VEHICLE TYPE CODE 1"]).size().reset_index(name="Count")
    if not trend_df.empty:
        fig_vehicle_trend = px.line(
            trend_df,
            x="YEAR",
            y="Count",
            color="VEHICLE TYPE CODE 1",
            labels={
                "YEAR": "Year", 
                "Count": "Number of Crashes", 
                "VEHICLE TYPE CODE 1": "Vehicle Type"
            },
            color_discrete_sequence=vibrant_colors
        )
        fig_vehicle_trend = apply_pink_template(fig_vehicle_trend)
        st.plotly_chart(fig_vehicle_trend, use_container_width=True)

# ===== TAB 3: PEOPLE & INJURIES =====
with tab3:
    # First row with Safety Equipment and Injury Types side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛡️ Safety Equipment")
        safety_dist = filtered_df.groupby("SAFETY_EQUIPMENT")["UNIQUE_ID"].count().reset_index(name="Count")
        safety_dist = safety_dist.sort_values("Count", ascending=False).head(5)
        if not safety_dist.empty:
            fig_safety = px.pie(
                safety_dist, 
                names="SAFETY_EQUIPMENT", 
                values="Count",
                labels={"SAFETY_EQUIPMENT": "Safety Equipment", "Count": "Number of Records"},
                color_discrete_sequence=['#FF8DA1', '#FFB6C1', '#FFD1DC', '#FFAEC9', '#FF85A1']
            )
            fig_safety = apply_pink_template(fig_safety)
            fig_safety.update_layout(margin=dict(t=40, b=20))
            st.plotly_chart(fig_safety, use_container_width=True)
    
    with col2:
        st.subheader("🚑 Injury Types")
        injury_df = filtered_df.groupby("BODILY_INJURY").size().reset_index(name="Count").sort_values("Count", ascending=False)
        if not injury_df.empty:
            fig_injury = px.bar(
                injury_df,
                x="Count",
                y="BODILY_INJURY",
                orientation="h",
                labels={"BODILY_INJURY": "Bodily Injury", "Count": "Number of Cases"},
                color_discrete_sequence=['#20B2AA']  # Teal color
            )
            fig_injury.update_yaxes(categoryorder="total ascending")
            fig_injury = apply_pink_template(fig_injury)
            fig_injury.update_layout(
                margin=dict(t=40, b=20),
                showlegend=False
            )
            st.plotly_chart(fig_injury, use_container_width=True)

# Continue with other tabs following the same pattern...

# Footer
st.markdown("---")
st.markdown("**NYC Crash Analysis Dashboard | Built with Streamlit & Plotly**")
