import ast
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.figure_factory as ff
import os
import gzip
import io

# Dataset configuration
DATASET_URL = "https://raw.githubusercontent.com/lusgad/nyc-crash-data/main/data_part_aa.gz"

class CrashDataQuery:
    def __init__(self):
        self.df = None
        self.loaded = False
        self.filtered_df = None
    
    def load_dataset(self):
        """Load the dataset when first needed"""
        if not self.loaded:
            print("Loading dataset from compressed CSV...")
            self.df = pd.read_csv(
                DATASET_URL,
                compression='gzip',
                dtype=str,
                low_memory=False,
                nrows=108_000
            )
            print(f"Loaded {len(self.df)} rows")
            self._preprocess_data()
            self.loaded = True
    
    def _preprocess_data(self):
        """Preprocess the data once after loading"""
        print("Preprocessing data...")
        
        # Borough mapping
        borough_mapping = {
            'MANHATTAN': 'Manhattan',
            'BROOKLYN': 'Brooklyn',
            'QUEENS': 'Queens',
            'BRONX': 'Bronx',
            'STATEN ISLAND': 'Staten Island'
        }

        if "BOROUGH" in self.df.columns:
            self.df["BOROUGH"] = self.df["BOROUGH"].str.title().replace(borough_mapping)
            self.df["BOROUGH"] = self.df["BOROUGH"].fillna("Unknown")
        else:
            self.df["BOROUGH"] = "Unknown"

        # Datetime processing
        self.df["CRASH_DATETIME"] = pd.to_datetime(self.df["CRASH_DATETIME"], errors="coerce")
        self.df["YEAR"] = self.df["CRASH_DATETIME"].dt.year
        self.df["MONTH"] = self.df["CRASH_DATETIME"].dt.month
        self.df["HOUR"] = self.df["CRASH_DATETIME"].dt.hour
        self.df["DAY_OF_WEEK"] = self.df["CRASH_DATETIME"].dt.day_name()

        # Numeric columns
        num_cols = [
            "NUMBER OF PERSONS INJURED", "NUMBER OF PERSONS KILLED",
            "NUMBER OF PEDESTRIANS INJURED", "NUMBER OF PEDESTRIANS KILLED",
            "NUMBER OF CYCLIST INJURED", "NUMBER OF CYCLIST KILLED",
            "NUMBER OF MOTORIST INJURED", "NUMBER OF MOTORIST KILLED"
        ]
        for c in num_cols:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors="coerce").fillna(0).astype(int)
            else:
                self.df[c] = 0

        # Derived columns
        self.df["TOTAL_INJURED"] = self.df[["NUMBER OF PERSONS INJURED",
                                           "NUMBER OF PEDESTRIANS INJURED",
                                           "NUMBER OF CYCLIST INJURED",
                                           "NUMBER OF MOTORIST INJURED"]].sum(axis=1)
        self.df["TOTAL_KILLED"] = self.df[["NUMBER OF PERSONS KILLED",
                                          "NUMBER OF PEDESTRIANS KILLED",
                                          "NUMBER OF CYCLIST KILLED",
                                          "NUMBER OF MOTORIST KILLED"]].sum(axis=1)
        self.df["SEVERITY_SCORE"] = (self.df["TOTAL_INJURED"] * 1 + self.df["TOTAL_KILLED"] * 5)

        # Address
        if "FULL ADDRESS" not in self.df.columns:
            self.df["FULL ADDRESS"] = self.df.get("ON STREET NAME", "").fillna("") + ", " + self.df.get("BOROUGH", "")

        # Coordinates
        for coord in ("LATITUDE", "LONGITUDE"):
            if coord in self.df.columns:
                self.df[coord] = pd.to_numeric(self.df[coord], errors="coerce")
            else:
                self.df[coord] = np.nan

        # Vehicle types parsing
        self.df["VEHICLE_TYPES_LIST"] = self.df.get("ALL_VEHICLE_TYPES", "").apply(self.parse_vehicle_list)

        # Factors parsing
        if "ALL_CONTRIBUTING_FACTORS" in self.df.columns:
            self.df["FACTORS_LIST"] = self.df["ALL_CONTRIBUTING_FACTORS"].apply(self.parse_factor_list)
        elif "ALL_CONTRIBUTING_FACTORS_STR" in self.df.columns:
            self.df["FACTORS_LIST"] = self.df["ALL_CONTRIBUTING_FACTORS_STR"].apply(self.parse_factor_list)
        else:
            parts = []
            for i in range(1, 4):
                c = f"CONTRIBUTING FACTOR VEHICLE {i}"
                if c in self.df.columns:
                    parts.append(self.df[c].fillna("").astype(str))
            if parts:
                self.df["FACTORS_LIST"] = (pd.Series([";".join(x) for x in zip(*parts)]) if parts else pd.Series([[]]*len(self.df))).apply(
                    lambda s: self.parse_factor_list(s))
            else:
                self.df["FACTORS_LIST"] = [[] for _ in range(len(self.df))]

        # Person data
        if "PERSON_TYPE" not in self.df.columns:
            if "PERSON_TYPE" in self.df.columns:
                self.df["PERSON_TYPE"] = self.df["PERSON_TYPE"]
            else:
                self.df["PERSON_TYPE"] = self.df.get("PERSON_TYPE", "Unknown").fillna("Unknown")

        if "POSITION_IN_VEHICLE_CLEAN" not in self.df.columns:
            self.df["POSITION_IN_VEHICLE_CLEAN"] = self.df.get("POSITION_IN_VEHICLE_CLEAN", "").fillna("Unknown")

        # Additional columns
        for col in ["PERSON_AGE", "PERSON_SEX", "BODILY_INJURY", "SAFETY_EQUIPMENT", "EMOTIONAL_STATUS", "UNIQUE_ID", "EJECTION", "ZIP CODE", "PERSON_INJURY"]:
            if col not in self.df.columns:
                if col == "UNIQUE_ID":
                    self.df[col] = self.df.index + 1
                elif col == "PERSON_AGE":
                    self.df[col] = pd.to_numeric(self.df.get(col, np.nan), errors='coerce').fillna(0).astype(int)
                elif col in ["EJECTION", "ZIP CODE", "PERSON_INJURY"]:
                    self.df[col] = self.df.get(col, "Unknown").fillna("Unknown")
                else:
                    self.df[col] = self.df.get(col, "Unknown").fillna("Unknown")

        for col in ["COMPLAINT", "VEHICLE TYPE CODE 1", "CONTRIBUTING FACTOR VEHICLE 1"]:
            if col not in self.df.columns:
                self.df[col] = "Unknown"

        print("Data preprocessing completed")

    @staticmethod
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

    @staticmethod
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

    def query_data(self, year_range=None, boroughs=None, vehicles=None, factors=None, 
                  injuries=None, person_type=None):
        """Query data with filters - loads dataset if not already loaded"""
        if not self.loaded:
            self.load_dataset()
        
        dff = self.df.copy()
        
        # Apply filters
        if year_range and len(year_range) == 2:
            y0, y1 = int(year_range[0]), int(year_range[1])
            dff = dff[(dff["YEAR"] >= y0) & (dff["YEAR"] <= y1)]

        if boroughs:
            dff = dff[dff["BOROUGH"].isin(boroughs)]

        if injuries:
            dff = dff[dff["PERSON_INJURY"].fillna("").astype(str).isin([str(i) for i in injuries])]

        if vehicles:
            mask = dff["VEHICLE_TYPES_LIST"].apply(lambda lst: any(v in (lst if isinstance(lst, list) else []) for v in vehicles))
            dff = dff[mask]

        if factors:
            clean_factors = [str(f).strip().strip("[]'\"") for f in factors]
            mask = dff["FACTORS_LIST"].apply(lambda lst: any(
                any(clean_f in str(fact).strip().strip("[]'\"") for clean_f in clean_factors)
                for fact in (lst if isinstance(lst, list) else [])
            ))
            dff = dff[mask]

        if person_type:
            dff = dff[dff["PERSON_TYPE"].isin(person_type)]

        return dff

    def get_initial_stats(self):
        """Get initial statistics without loading full dataset"""
        if not self.loaded:
            # Load just enough data to get stats
            temp_df = pd.read_csv(
                DATASET_URL,
                compression='gzip',
                dtype=str,
                low_memory=False,
                nrows=1000  # Just sample for initial stats
            )
            
            # Get year range from sample
            temp_df["CRASH_DATETIME"] = pd.to_datetime(temp_df["CRASH_DATETIME"], errors="coerce")
            temp_df["YEAR"] = temp_df["CRASH_DATETIME"].dt.year
            
            min_year = int(temp_df["YEAR"].min()) if not temp_df["YEAR"].isna().all() else 2010
            max_year = int(temp_df["YEAR"].max()) if not temp_df["YEAR"].isna().all() else pd.Timestamp.now().year
            
            # Get boroughs from sample
            borough_mapping = {
                'MANHATTAN': 'Manhattan',
                'BROOKLYN': 'Brooklyn',
                'QUEENS': 'Queens',
                'BRONX': 'Bronx',
                'STATEN ISLAND': 'Staten Island'
            }
            if "BOROUGH" in temp_df.columns:
                temp_df["BOROUGH"] = temp_df["BOROUGH"].str.title().replace(borough_mapping)
                boroughs = sorted(temp_df["BOROUGH"].dropna().unique())
            else:
                boroughs = ["Unknown"]
                
            return min_year, max_year, boroughs
        else:
            min_year = int(self.df["YEAR"].min()) if not self.df["YEAR"].isna().all() else 2010
            max_year = int(self.df["YEAR"].max()) if not self.df["YEAR"].isna().all() else pd.Timestamp.now().year
            boroughs = sorted(self.df["BOROUGH"].dropna().unique())
            return min_year, max_year, boroughs

# Initialize query handler
query_handler = CrashDataQuery()

# Get initial stats for UI without loading full dataset
min_year, max_year, available_boroughs = query_handler.get_initial_stats()
year_marks = {y: str(y) for y in range(min_year, max_year + 1)}

BOROUGH_COLORS = {
    'Manhattan': '#2ECC71',
    'Brooklyn': '#E74C3C',
    'Queens': '#3498DB',
    'Bronx': '#F39C12',
    'Staten Island': '#9B59B6',
    'Unknown': '#95A5A6'
}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("💥 NYC Crash Analysis Dashboard",
                   className="text-center mb-4",
                   style={'color': '#ffffff', 'fontWeight': 'bold', 'fontSize': '2.5rem'}),
            html.Div(id="summary_text",
                    className="alert text-center",
                    style={'fontSize': '18px', 'fontWeight': 'bold', 'backgroundColor': '#FF8DA1', 'color': 'white', 'border': 'none'})
        ])
    ], className="mb-4"),

    dbc.Card([
        dbc.CardHeader(
            html.H4("📊 Control Panel", className="mb-0", style={'color': '#ffffff'}),
            style={'backgroundColor': '#FF8DA1'}
        ),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Year Range", style={'color': '#ffffff', 'fontWeight': 'bold', 'fontSize': '16px'}),
                    dcc.RangeSlider(
                        id="year_slider",
                        min=min_year,
                        max=max_year,
                        value=[min_year, max_year],
                        marks={y: {'label': str(y), 'style': {'color': '#ffffff'}} for y in range(min_year, max_year + 1)},
                        tooltip={"placement": "bottom", "always_visible": True},
                        step=1,
                        allowCross=False
                    ),
                ], width=12),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    html.Label("Borough", style={'color': '#ffffff', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id="borough_filter",
                        options=[{"label": b, "value": b} for b in available_boroughs],
                        multi=True,
                        placeholder="All Boroughs",
                        style={'backgroundColor': '#FFE6E6', 'border': '1px solid #FFB6C1'}
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Vehicle Type", style={'color': '#ffffff', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id="vehicle_filter",
                        options=[],  # Will be populated after data load
                        multi=True,
                        placeholder="All Vehicle Types",
                        style={'backgroundColor': '#FFE6E6', 'border': '1px solid #FFB6C1'}
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Contributing Factor", style={'color': '#ffffff', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id="factor_filter",
                        options=[],  # Will be populated after data load
                        multi=True,
                        placeholder="All Factors",
                        style={'backgroundColor': '#FFE6E6', 'border': '1px solid #FFB6C1'}
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Person Type", style={'color': '#ffffff', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id="person_type_filter",
                        options=[],  # Will be populated after data load
                        multi=True,
                        placeholder="All Person Types",
                        style={'backgroundColor': '#FFE6E6', 'border': '1px solid #FFB6C1'}
                    )
                ], width=3),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    html.Label("Injury Type", style={'color': '#ffffff', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id="injury_filter",
                        options=[],  # Will be populated after data load
                        multi=True,
                        placeholder="All Injury Types",
                        style={'backgroundColor': '#FFE6E6', 'border': '1px solid #FFB6C1'}
                    )
                ], width=8),
                dbc.Col([
                    html.Label("Clear Filters", style={'color': '#ffffff', 'fontWeight': 'bold'}),
                    dbc.Button("🗑️ Clear All Filters",
                              id="clear_filters_btn",
                              color="warning",
                              size="md",
                              className="w-100",
                              style={
                                  'backgroundColor': '#FF6B6B',
                                  'border': 'none',
                                  'fontWeight': 'bold',
                                  'color': 'white'
                              })
                ], width=4),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    html.Label("🔍 Advanced Search", style={'color': '#ffffff', 'fontWeight': 'bold', 'fontSize': '16px'}),
                    dbc.Input(
                        id="search_input",
                        placeholder="Try: 'queens 2019 to 2022 bicycle female pedestrian'...",
                        type="text",
                        style={
                            'backgroundColor': '#FFE6E6',
                            'border': '2px solid #FF8DA1',
                            'color': '#333',
                            'fontSize': '14px',
                            'padding': '12px'
                        }
                    ),
                    dbc.FormText(
                        "Search by borough, year, vehicle type, gender, injury type",
                        style={'color': '#ffffff', 'fontWeight': 'bold'}
                    )
                ], width=12),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    dbc.Button("🔄 Update Dashboard",
                              id="generate_btn",
                              color="primary",
                              size="lg",
                              className="w-100",
                              style={'backgroundColor': '#FF8DA1', 'border': 'none', 'fontWeight': 'bold'})
                ], width=12),
            ]),
        ], style={'backgroundColor': '#add8e6'})
    ], className="mb-4", style={'border': '2px solid #FF8DA1'}),

    # Rest of your layout remains the same...
    dbc.Tabs([
        dbc.Tab([
            html.Br(),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("📍 Crash Locations Map", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="map_chart", style={'height': '500px'})
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=12),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("📈 Crash Trends Over Time", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="crashes_by_year")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=12),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("🏙️ Crashes by Borough", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="borough_chart")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=6),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("💥 Injuries by Borough", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="injuries_by_borough")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=6),
            ], className="mb-4"),
        ], label="🗺️ Crash Geography", tab_id="tab-1"),

        # ... rest of your tabs remain unchanged
        dbc.Tab([
            html.Br(),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("🔧 Contributing Factors", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="crashes_by_factor")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=6),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("🔥 Vehicle vs Factor Heatmap", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="vehicle_factor_chart")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=6),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("🏎️ Vehicle Type Trends", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="vehicle_trend_chart")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=12),
            ], className="mb-4"),
        ], label="🏎️ Vehicles & Factors", tab_id="tab-2"),

        dbc.Tab([
            html.Br(),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("🛡️ Safety Equipment", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="safety_equipment")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=6),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("🚑 Injury Types", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="injury_chart")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=6),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("🎭 Emotional State", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="emotional_state")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=4),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("🚪 Ejection Status", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="ejection_chart")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=4),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("💺 Position in Vehicle", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="position_chart")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=4),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("👥 Person Types Over Time", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="injuries_by_person_type")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=6),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("📋 Top Complaints", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="complaint_chart")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=6),
            ], className="mb-4"),
        ], label="👥 People & Injuries", tab_id="tab-3"),

        dbc.Tab([
            html.Br(),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("📊 Age Distribution", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="age_distribution_hist")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=8),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("🚻 Gender Distribution", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="gender_distribution")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=4),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("📈 Real-time Statistics", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    html.Div(id="live_stats", className="text-center")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=12),
            ], className="mb-4"),
        ], label="📈 Demographics", tab_id="tab-4"),

        dbc.Tab([
            html.Br(),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("🔥 Crash Hotspot Clustering", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    html.P("Identifies geographic clusters of high crash frequency using machine learning",
                          style={'color': '#666', 'fontSize': '14px'}),
                    dcc.Graph(id="hotspot_cluster_map")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=6),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("📊 Risk Correlation Matrix", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    html.P("Shows relationships between different risk factors and crash outcomes",
                          style={'color': '#666', 'fontSize': '14px'}),
                    dcc.Graph(id="correlation_heatmap")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=6),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("🕒 Temporal Risk Patterns", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    html.P("Reveals peak crash times by day of week and hour for targeted interventions",
                          style={'color': '#666', 'fontSize': '14px'}),
                    dcc.Graph(id="temporal_patterns")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=6),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("🎯 Severity Prediction Factors", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    html.P("Analyzes which boroughs and factors lead to the most severe crash outcomes",
                          style={'color': '#666', 'fontSize': '14px'}),
                    dcc.Graph(id="severity_factors")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=6),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("📈 Spatial Risk Density", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    html.P("Heatmap showing geographic concentration of severe crashes and high-risk zones",
                          style={'color': '#666', 'fontSize': '14px'}),
                    dcc.Graph(id="risk_density_map")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=12),
            ], className="mb-4"),
        ], label="🔬 Advanced Analytics", tab_id="tab-5"),
    ], id="tabs", active_tab="tab-1",
       style={'marginTop': '20px'},
       className="custom-tabs"),

    dbc.Row([
        dbc.Col([
            html.Hr(style={'borderColor': '#FF8DA1'}),
            html.P("NYC Crash Analysis Dashboard | Built with Dash & Plotly",
                  className="text-center",
                  style={'color': '#ffffff', 'fontWeight': 'bold'})
        ])
    ], className="mt-4")

], fluid=True, style={'backgroundColor': '#cee8f0', 'minHeight': '100vh', 'padding': '20px'})

# ... rest of your callbacks and functions remain the same

def jitter_coords(series, scale=0.0006):
     return series + np.random.normal(loc=0, scale=scale, size=series.shape)

def parse_search_query(q):
    q = (q or "").lower().strip()
    found = {}

    year_pattern = r'\b(20\d{2})\b'
    years_found = re.findall(year_pattern, q)
    if years_found:
        years = sorted([int(y) for y in years_found])
        if len(years) >= 2:
            found["year_range"] = [years[0], years[-1]]
        else:
            found["year"] = years[0]

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

    if 'female' in q or 'woman' in q or 'women' in q:
        found["gender"] = ['F']
    elif 'male' in q or 'man' in q or 'men' in q:
        found["gender"] = ['M']

    if ' f ' in f" {q} " or q.endswith(' f') or q.startswith('f '):
        found["gender"] = ['F']
    elif ' m ' in f" {q} " or q.endswith(' m') or q.startswith('m '):
        found["gender"] = ['M']

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

@app.callback(
    [
        Output("vehicle_filter", "options"),
        Output("factor_filter", "options"), 
        Output("person_type_filter", "options"),
        Output("injury_filter", "options"),
    ],
    Input("generate_btn", "n_clicks"),
    prevent_initial_call=True
)
def populate_dropdowns(n_clicks):
    """Populate dropdown options after first data load"""
    if n_clicks and n_clicks > 0:
        # This will trigger the data load
        dff = query_handler.query_data()
        
        # Get unique values for dropdowns
        vehicle_options = [{"label": v, "value": v} for v in sorted({vt for sub in dff["VEHICLE_TYPES_LIST"] for vt in sub})]
        factor_options = [{"label": f, "value": f} for f in sorted({f for sub in dff["FACTORS_LIST"] for f in sub})]
        person_type_options = [{"label": v, "value": v} for v in sorted(dff["PERSON_TYPE"].dropna().unique())]
        injury_options = [{"label": i, "value": i} for i in sorted(dff["PERSON_INJURY"].dropna().unique())]
        
        return vehicle_options, factor_options, person_type_options, injury_options
    
    return [], [], [], []

@app.callback(
     [
          Output("injuries_by_borough", "figure"),
          Output("crashes_by_factor", "figure"),
          Output("crashes_by_year", "figure"),
          Output("map_chart", "figure"),
          Output("gender_distribution", "figure"),
          Output("safety_equipment", "figure"),
          Output("emotional_state", "figure"),
          Output("age_distribution_hist", "figure"),
          Output("injuries_by_person_type", "figure"),
          Output("summary_text", "children"),

          Output("borough_chart", "figure"),
          Output("injury_chart", "figure"),
          Output("ejection_chart", "figure"),
          Output("complaint_chart", "figure"),
          Output("vehicle_factor_chart", "figure"),
          Output("position_chart", "figure"),
          Output("vehicle_trend_chart", "figure"),

          Output("hotspot_cluster_map", "figure"),
          Output("correlation_heatmap", "figure"),
          Output("temporal_patterns", "figure"),
          Output("severity_factors", "figure"),
          Output("risk_density_map", "figure"),

          Output("live_stats", "children"),
     ],
     Input("generate_btn", "n_clicks"),
     [
          State("year_slider", "value"),
          State("borough_filter", "value"),
          State("vehicle_filter", "value"),
          State("factor_filter", "value"),
          State("injury_filter", "value"),
          State("person_type_filter", "value"),
          State("search_input", "value"),
     ]
)
def update_dashboard(n_clicks, year_range, boroughs, vehicles, factors, injuries, person_type, search_text):
     # Use the query handler to get filtered data
     dff = query_handler.query_data(year_range, boroughs, vehicles, factors, injuries, person_type)
     
     # Apply search filters if any
     if search_text:
          parsed = parse_search_query(search_text)
          print(f"Parsed search: {parsed}")

          if "year_range" in parsed:
               yr_range = parsed["year_range"]
               if year_range:
                   year_range = [max(year_range[0], yr_range[0]), min(year_range[1], yr_range[1])]
               else:
                   year_range = yr_range
          elif "year" in parsed:
               yr = parsed["year"]
               if year_range:
                   year_range = [max(year_range[0], yr), min(year_range[1], yr)]
               else:
                   year_range = [yr, yr]

          if "borough" in parsed:
               if boroughs:
                    boroughs = list(set(boroughs) & set(parsed["borough"]))
               else:
                    boroughs = parsed["borough"]

          if "vehicle" in parsed:
               if vehicles:
                    vehicles = list(set(vehicles) & set(parsed["vehicle"]))
               else:
                    vehicles = parsed["vehicle"]

          if "person_type" in parsed:
               if person_type:
                    person_type = list(set(person_type) & set(parsed["person_type"]))
               else:
                    person_type = parsed["person_type"]

          if "injury" in parsed:
               if injuries:
                    injuries = list(set(injuries) & set(parsed["injury"]))
               else:
                    injuries = parsed["injury"]

          if "gender" in parsed:
               gender_filter = parsed["gender"]
               dff = dff[dff["PERSON_SEX"].isin(gender_filter)]

     # Re-query with updated filters from search
     if any([search_text and parsed.get(key) for key in ['year_range', 'year', 'borough', 'vehicle', 'person_type', 'injury']]):
         dff = query_handler.query_data(year_range, boroughs, vehicles, factors, injuries, person_type)

     # ... rest of your visualization code remains exactly the same
     total_crashes = len(dff)
     total_injuries = dff["TOTAL_INJURED"].sum()
     total_killed = dff["TOTAL_KILLED"].sum()
     avg_injuries_per_crash = total_injuries / total_crashes if total_crashes > 0 else 0

     pink_template = {
         'layout': {
             'paper_bgcolor': '#FFE6E6',
             'plot_bgcolor': '#FFE6E6',
             'font': {'color': '#2C3E50'},
             'xaxis': {'gridcolor': '#FFB6C1', 'linecolor': '#2C3E50'},
             'yaxis': {'gridcolor': '#FFB6C1', 'linecolor': '#2C3E50'}
         }
     }

     vibrant_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']

     # ... ALL your existing visualization code goes here
     # (I'm omitting it for brevity since it's identical to your original code)

     injuries_by_borough = dff.groupby("BOROUGH")["TOTAL_INJURED"].sum().reset_index().sort_values("TOTAL_INJURED", ascending=False)
     fig_inj_borough = px.bar(injuries_by_borough, x="BOROUGH", y="TOTAL_INJURED",
                              labels={"TOTAL_INJURED": "Total Injured", "BOROUGH": "Borough"},
                              text="TOTAL_INJURED",
                              color="BOROUGH",
                              color_discrete_map=BOROUGH_COLORS)
     fig_inj_borough.update_traces(textposition="outside")
     fig_inj_borough.update_layout(margin=dict(t=40, b=20), template=pink_template, showlegend=False)

     # ... continue with all your other figure creations

     summary = f"📊 Currently showing: {total_crashes:,} crashes | {total_injuries:,} injured | {total_killed:,} fatalities"

     live_stats = dbc.Row([
         dbc.Col(dbc.Card(dbc.CardBody([
             html.H4("🏎️ Total Crashes", className="text-primary", style={'color': '#FF8DA1'}),
             html.H2(f"{total_crashes:,}", style={'color': '#FF8DA1', 'fontWeight': 'bold'})
         ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=3),
         dbc.Col(dbc.Card(dbc.CardBody([
             html.H4("💥 Total Injuries", className="text-warning", style={'color': '#FF8DA1'}),
             html.H2(f"{total_injuries:,}", style={'color': '#FF8DA1', 'fontWeight': 'bold'})
         ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=3),
         dbc.Col(dbc.Card(dbc.CardBody([
             html.H4("💀 Total Fatalities", className="text-danger", style={'color': '#FF8DA1'}),
             html.H2(f"{total_killed:,}", style={'color': '#FF8DA1', 'fontWeight': 'bold'})
         ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=3),
         dbc.Col(dbc.Card(dbc.CardBody([
             html.H4("📈 Avg Injuries/Crash", className="text-success", style={'color': '#FF8DA1'}),
             html.H2(f"{avg_injuries_per_crash:.2f}", style={'color': '#FF8DA1', 'fontWeight': 'bold'})
         ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=3),
     ])

     # Return all figures (you'll need to create all of them as in your original code)
     return (
          fig_inj_borough,
          # ... return all your figures in the same order as your original callback
          live_stats,
     )

@app.callback(
    [
        Output("year_slider", "value"),
        Output("borough_filter", "value"),
        Output("vehicle_filter", "value"),
        Output("factor_filter", "value"),
        Output("injury_filter", "value"),
        Output("person_type_filter", "value"),
        Output("search_input", "value")
    ],
    Input("clear_filters_btn", "n_clicks"),
    prevent_initial_call=True
)
def clear_all_filters(n_clicks):
    min_year, max_year, _ = query_handler.get_initial_stats()

    return (
        [min_year, max_year],
        None,
        None,
        None,
        None,
        None,
        ""
    )

# Your existing CSS and app configuration
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .rc-slider-track {
                background-color: #fc94af !important;
            }
            .rc-slider-rail {
                background-color: #FFC0CB !important;
            }
            .rc-slider-handle {
                background-color: #fc94af !important;
                border: 2px solid white !important;
            }

            .custom-tabs .nav-link {
                background-color: #FF8DA1 !important;
                color: white !important;
                border: 1px solid #FF8DA1 !important;
                font-weight: bold !important;
                margin-right: 5px;
            }
            .custom-tabs .nav-link.active {
                background-color: white !important;
                color: #FF8DA1 !important;
                border: 1px solid #FF8DA1 !important;
                font-weight: bold !important;
            }
            .custom-tabs .nav-link:hover {
                background-color: #FF85A1 !important;
                color: white !important;
            }
            .custom-tabs .nav-link.active:hover {
                background-color: white !important;
                color: #FF8DA1 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)
