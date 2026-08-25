import os
import gzip
import io
import zipfile
import datetime
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import fitparse
import gpxpy
import folium
from folium.features import ColorLine
from branca.colormap import LinearColormap
from streamlit_folium import st_folium
from garminconnect import Garmin
from dotenv import load_dotenv

# Load environment variables from .env file if present locally
load_dotenv()

st.set_page_config(
    page_title="Fusion Metrics | Fitness Telemetry Engine",
    page_icon="⚡",
    layout="wide"
)

# Helper function to read secrets locally (.env / os.getenv) or on Streamlit Cloud (st.secrets)
def get_secret(key_name, default=""):
    if hasattr(st, "secrets") and key_name in st.secrets:
        return st.secrets[key_name]
    return os.getenv(key_name, default)

# Helper function to format seconds into hh:mm:ss
def format_seconds(seconds):
    try:
        s = int(seconds or 0)
        hrs = s // 3600
        mins = (s % 3600) // 60
        secs = s % 60
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    except:
        return "00:00:00"

# -------------------------------------------------------------------
# FUSION METRICS Custom CSS & Theme Styling
# -------------------------------------------------------------------
st.markdown("""
    <style>
    /* Dark Theme Core Backgrounds */
    .stApp {
        background-color: #0B132B;
        color: #E0E1DD;
    }
    
    /* Sidebar Dark Styling */
    [data-testid="stSidebar"] {
        background-color: #070D1E;
        border-right: 1px solid #1C2541;
    }
    
    /* Metric Cards Custom Accent */
    div[data-testid="stMetricValue"] {
        color: #00B4D8 !important;
        font-weight: 700;
    }
    
    /* Buttons with Brand Gradient */
    .stButton>button {
        background: linear-gradient(135deg, #0077B6 0%, #00B4D8 50%, #70E000 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        opacity: 0.9 !important;
        transform: translateY(-1px);
    }
    
    /* Active Tab Line Styling */
    button[data-baseweb="tab"] {
        color: #8D99AE !important;
    }
    button[aria-selected="true"] {
        color: #00B4D8 !important;
        border-bottom-color: #70E000 !important;
    }
    </style>
""", unsafe_allow_html=True)

DB_PATH = "fitness.db"

METER_BASED_SPORTS = [
    "rowing", "indoor_rowing", "virtual_rowing", 
    "swimming", "lap_swimming", "kayaking", "canoeing", 
    "paddling", "stand_up_paddleboarding"
]

def is_meter_based(activity_type_str):
    if not activity_type_str:
        return False
    clean_type = str(activity_type_str).lower().replace(" ", "_")
    return any(sport in clean_type for sport in METER_BASED_SPORTS)

def get_activity_emoji(activity_type_str):
    if not activity_type_str:
        return "⚡"
    act_clean = str(activity_type_str).lower().replace("_", " ").strip()
    if any(s in act_clean for s in ["snowboard", "snowboarding"]):
        return "🏂"
    elif any(s in act_clean for s in ["ski", "skiing", "resort skiing", "alpine skiing"]):
        return "🎿"
    elif any(c in act_clean for c in ["ride", "bike", "cycling", "e bike", "gravel"]):
        return "🚴"
    elif "run" in act_clean or "running" in act_clean:
        return "🏃"
    elif any(h in act_clean for h in ["trail", "hike", "hiking"]):
        return "🥾"
    elif "walk" in act_clean or "walking" in act_clean:
        return "🚶"
    elif "swim" in act_clean or "swimming" in act_clean:
        return "🏊"
    elif any(w in act_clean for w in ["rowing", "paddling", "kayaking", "canoeing", "sup"]):
        return "🚣"
    elif "softball" in act_clean or "baseball" in act_clean:
        return "⚾"
    elif any(g in act_clean for g in ["strength", "weight", "gym", "workout"]):
        return "🏋️"
    return "⚡"

# -------------------------------------------------------------------
# Database Initialization & Schema
# -------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS activities (
            activity_id TEXT PRIMARY KEY,
            provider TEXT,
            activity_date TEXT,
            activity_name TEXT,
            activity_type TEXT,
            distance_mi REAL,
            moving_time_sec REAL,
            elapsed_time_sec REAL,
            max_speed_mph REAL,
            avg_speed_mph REAL,
            max_hr INTEGER,
            avg_hr INTEGER,
            total_ascent_ft REAL,
            total_descent_ft REAL,
            start_lat REAL,
            start_lon REAL,
            end_lat REAL,
            end_lon REAL,
            route_id INTEGER DEFAULT -1,
            flagged INTEGER DEFAULT 0,
            filename TEXT
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_act_date ON activities(activity_date)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_act_route ON activities(route_id)")
    conn.commit()
    conn.close()

init_db()

# -------------------------------------------------------------------
# Matched Routes Clustering Algorithm
# -------------------------------------------------------------------
def update_route_clusters(dist_threshold_miles=0.25):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT activity_id, start_lat, start_lon, end_lat, end_lon, distance_mi, route_id FROM activities", conn)
    
    if df.empty:
        conn.close()
        return

    df["new_route_id"] = -1
    valid = df.dropna(subset=["start_lat", "start_lon", "end_lat", "end_lon", "distance_mi"]).copy()
    
    route_counter = 1
    for idx, row in valid.iterrows():
        if valid.loc[idx, "new_route_id"] != -1:
            continue
            
        st_lat, st_lon = row["start_lat"], row["start_lon"]
        en_lat, en_lon = row["end_lat"], row["end_lon"]
        dist = row["distance_mi"]

        lat_tol = dist_threshold_miles / 69.0
        lon_tol = dist_threshold_miles / (69.0 * np.cos(np.radians(st_lat)))

        matches = valid[
            (valid["new_route_id"] == -1) &
            (np.abs(valid["start_lat"] - st_lat) <= lat_tol) &
            (np.abs(valid["start_lon"] - st_lon) <= lon_tol) &
            (np.abs(valid["end_lat"] - en_lat) <= lat_tol) &
            (np.abs(valid["end_lon"] - en_lon) <= lon_tol) &
            (np.abs(valid["distance_mi"] - dist) <= (dist * 0.08))
        ]

        if len(matches) >= 2:
            valid.loc[matches.index, "new_route_id"] = route_counter
            route_counter += 1

    c = conn.cursor()
    for idx, row in valid.iterrows():
        c.execute("UPDATE activities SET route_id = ? WHERE activity_id = ?", (int(row["new_route_id"]), str(row["activity_id"])))
    conn.commit()
    conn.close()

# -------------------------------------------------------------------
# File Parsers & Data Normalization
# -------------------------------------------------------------------
def convert_semicircles(s):
    return s * (180 / (2**31)) if s is not None and not pd.isna(s) else None

@st.cache_data
def parse_fit_file(filepath):
    try:
        f_obj = gzip.open(filepath, "rb") if filepath.endswith(".gz") else open(filepath, "rb")
        fitfile = fitparse.FitFile(io.BytesIO(f_obj.read()) if filepath.endswith(".gz") else filepath)
        records = []
        for record in fitfile.get_messages("record"):
            data = {r.name: r.value for r in record}
            records.append(data)
        df = pd.DataFrame(records)
        if df.empty:
            return df
        if "position_lat" in df.columns:
            df["lat"] = df["position_lat"].apply(convert_semicircles)
            df["lon"] = df["position_long"].apply(convert_semicircles)
        if "speed" in df.columns:
            df["Speed (mph)"] = df["speed"] * 2.23694
        if "altitude" in df.columns:
            df["Elevation (ft)"] = df["altitude"] * 3.28084
        df = df.rename(columns={"heart_rate": "Heart Rate (bpm)", "cadence": "Cadence (rpm)", "timestamp": "Time"})
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data
def parse_gpx_file(filepath):
    try:
        f = gzip.open(filepath, "rt", encoding="utf-8") if filepath.endswith(".gz") else open(filepath, "r", encoding="utf-8")
        gpx = gpxpy.parse(f)
        data = []
        for track in gpx.tracks:
            for seg in track.segments:
                for pt in seg.points:
                    data.append({"Time": pt.time, "lat": pt.latitude, "lon": pt.longitude, "Elevation (ft)": (pt.elevation or 0) * 3.28084})
        return pd.DataFrame(data)
    except Exception as e:
        return pd.DataFrame()

def ingest_local_csv_to_sqlite():
    if not os.path.exists("activities.csv"):
        return
    df = pd.read_csv("activities.csv")
    df["Activity Date"] = pd.to_datetime(df["Activity Date"], errors="coerce")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    for idx, row in df.iterrows():
        act_id = f"local_{row['Activity Date'].strftime('%Y%m%d_%H%M%S')}"
        raw_dist = float(str(row.get("Distance", 0)).replace(",", ""))
        dist_mi = raw_dist * 0.000621371 if raw_dist > 1000 else raw_dist
        mov_sec = float(str(row.get("Moving Time", 0)).replace(",", ""))
        if mov_sec < 300:
            mov_sec *= 60

        c.execute("""
            INSERT INTO activities (
                activity_id, provider, activity_date, activity_name, activity_type,
                distance_mi, moving_time_sec, max_speed_mph, avg_speed_mph,
                max_hr, avg_hr, total_ascent_ft, filename
            ) VALUES (?, 'local_csv', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(activity_id) DO NOTHING
        """, (
            act_id, str(row["Activity Date"]), row.get("Activity Name", "Workout"),
            row.get("Activity Type", "Ride"), dist_mi, mov_sec,
            float(row.get("Max Speed", 0)), float(row.get("Average Speed", 0)),
            int(row.get("Max Heart Rate", 0)), int(row.get("Average Heart Rate", 0)),
            float(row.get("Elevation Gain", 0)), str(row.get("Filename", ""))
        ))
    conn.commit()
    conn.close()

ingest_local_csv_to_sqlite()

# -------------------------------------------------------------------
# Sidebar Brand Header & Workspace Controls
# -------------------------------------------------------------------
st.sidebar.markdown("""
    <div style="text-align: center; padding: 10px 0 20px 0;">
        <svg width="60" height="60" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M50 10L15 85H32L50 45L68 85H85L50 10Z" fill="url(#brand_grad_1)"/>
            <path d="M30 65C40 60 50 70 60 65C65 62.5 70 55 75 50" stroke="url(#brand_grad_2)" stroke-width="6" stroke-linecap="round"/>
            <path d="M22 78C35 72 48 82 62 76C70 72 75 65 80 60" stroke="#00B4D8" stroke-width="5" stroke-linecap="round"/>
            <defs>
                <linearGradient id="brand_grad_1" x1="50" y1="10" x2="50" y2="85" gradientUnits="userSpaceOnUse">
                    <stop stop-color="#70E000"/>
                    <stop offset="0.5" stop-color="#00B4D8"/>
                    <stop offset="1" stop-color="#0077B6"/>
                </linearGradient>
                <linearGradient id="brand_grad_2" x1="30" y1="65" x2="75" y2="50" gradientUnits="userSpaceOnUse">
                    <stop stop-color="#00B4D8"/>
                    <stop offset="1" stop-color="#70E000"/>
                </linearGradient>
            </defs>
        </svg>
        <h2 style="color: #FFFFFF; font-family: 'Helvetica Neue', sans-serif; font-weight: 800; letter-spacing: 2px; margin: 5px 0 0 0; font-size: 22px;">FUSION</h2>
        <h4 style="color: #00B4D8; font-family: 'Helvetica Neue', sans-serif; font-weight: 400; letter-spacing: 4px; margin: 0; font-size: 11px;">METRICS</h4>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🌐 Data Workspace")
data_source = st.sidebar.radio(
    "Data Source", 
    ["Local Offline Archive", "Garmin Connect (Live)"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Date Scope")

default_start = datetime.date(2024, 1, 1)
default_end = datetime.date.today()

if "date_range_picker" not in st.session_state:
    st.session_state["date_range_picker"] = (default_start, default_end)

pcol1, pcol2, pcol3 = st.sidebar.columns(3)

if pcol1.button("30 Days", use_container_width=True):
    st.session_state["date_range_picker"] = (datetime.date.today() - datetime.timedelta(days=30), datetime.date.today())
    st.rerun()

if pcol2.button("YTD", use_container_width=True):
    st.session_state["date_range_picker"] = (datetime.date(datetime.date.today().year, 1, 1), datetime.date.today())
    st.rerun()

if pcol3.button("All Time", use_container_width=True):
    st.session_state["date_range_picker"] = (datetime.date(2015, 1, 1), datetime.date.today())
    st.rerun()

date_range = st.sidebar.date_input("Select Range", key="date_range_picker", label_visibility="collapsed")

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range[0], date_range[1]
else:
    start_date, end_date = default_start, default_end

# -------------------------------------------------------------------
# Garmin Live Sync Button Execution Handler
# -------------------------------------------------------------------
if data_source == "Garmin Connect (Live)":
    st.sidebar.markdown("---")
    default_email = get_secret("GARMIN_EMAIL")
    default_password = get_secret("GARMIN_PASSWORD")

    with st.sidebar.expander("🔑 Garmin Credentials", expanded=not bool(default_email)):
        g_email = st.text_input("Email", value=default_email, autocomplete="username")
        g_password = st.text_input("Password", value=default_password, type="password", autocomplete="current-password")

    force_refresh = st.sidebar.button("🔄 Sync Live", type="primary", use_container_width=True)

    if force_refresh:
        if not g_email or not g_password:
            st.sidebar.error("Please provide your Garmin email and password.")
        else:
            with st.spinner("Authenticating and downloading Garmin activities..."):
                try:
                    # Initialize Garmin client
                    garmin = Garmin(g_email, g_password)
                    garmin.login()
                    
                    # Fetch latest 40 activities
                    activities = garmin.get_activities(0, 40)
                    
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    
                    added_count = 0
                    for act in activities:
                        act_id = f"garmin_{act['activityId']}"
                        act_date = act.get("startTimeLocal", "")
                        act_name = act.get("activityName", "Garmin Activity")
                        act_type = act.get("activityType", {}).get("typeKey", "workout")
                        dist_mi = (act.get("distance", 0) or 0) * 0.000621371
                        mov_sec = act.get("duration", 0) or 0
                        avg_spd = (act.get("averageSpeed", 0) or 0) * 2.23694
                        max_spd = (act.get("maxSpeed", 0) or 0) * 2.23694
                        avg_hr = act.get("averageHR", 0) or 0
                        max_hr = act.get("maxHR", 0) or 0
                        st_lat = act.get("startLatitude")
                        st_lon = act.get("startLongitude")
                        end_lat = act.get("endLatitude")
                        end_lon = act.get("endLongitude")

                        c.execute("""
                            INSERT INTO activities (
                                activity_id, provider, activity_date, activity_name, activity_type,
                                distance_mi, moving_time_sec, max_speed_mph, avg_speed_mph,
                                max_hr, avg_hr, start_lat, start_lon, end_lat, end_lon
                            ) VALUES (?, 'garmin_live', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(activity_id) DO UPDATE SET
                                activity_name=excluded.activity_name,
                                avg_speed_mph=excluded.avg_speed_mph,
                                distance_mi=excluded.distance_mi,
                                moving_time_sec=excluded.moving_time_sec
                        """, (act_id, act_date, act_name, act_type, dist_mi, mov_sec, max_spd, avg_spd, max_hr, avg_hr, st_lat, st_lon, end_lat, end_lon))
                        added_count += 1

                    conn.commit()
                    conn.close()
                    
                    # Update matched route clusters with newly ingested activities
                    update_route_clusters()
                    
                    st.sidebar.success(f"Successfully ingested {added_count} activities!")
                    st.rerun()

                except Exception as e:
                    st.sidebar.error(f"Garmin Sync Error: {str(e)}")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Re-cluster Matched Routes", use_container_width=True):
    with st.spinner("Clustering matched rides..."):
        update_route_clusters()
    st.sidebar.success("Route clustering complete!")

show_flagged = st.sidebar.checkbox("🚩 Show Flagged Activities", value=False)

# -------------------------------------------------------------------
# Main Dashboard Query & Header
# -------------------------------------------------------------------
st.title("⚡ Fusion Metrics Telemetry & Route Intelligence")

conn = sqlite3.connect(DB_PATH)
flag_clause = "" if show_flagged else "AND flagged = 0"
query = f"""
    SELECT *,
           moving_time_sec / 60.0 AS "Moving Time (min)"
    FROM activities
    WHERE DATE(activity_date) >= ? AND DATE(activity_date) <= ? {flag_clause}
    ORDER BY activity_date DESC
"""
df_all = pd.read_sql_query(query, conn, params=(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))
conn.close()

if df_all.empty:
    st.info("No activities found for selected range.")
    st.stop()

df_all["Activity Date"] = pd.to_datetime(df_all["activity_date"])
df_all["Display Label"] = df_all.apply(lambda r: f"{'🚩 [FLAGGED] ' if r['flagged']==1 else ''}{r['Activity Date'].strftime('%Y-%m-%d %H:%M')} - {r['activity_name']}", axis=1)

# Summary Top Bar
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Activities", len(df_all))
col2.metric("Total Distance", f"{df_all['distance_mi'].sum():,.1f} mi")
col3.metric("Total Moving Time", f"{(df_all['moving_time_sec'].sum() / 3600):,.1f} hrs")
col4.metric("Top Speed", f"{df_all['max_speed_mph'].max():.1f} mph" if df_all['max_speed_mph'].max() > 0 else "N/A")

st.markdown("---")

# -------------------------------------------------------------------
# Views Switcher
# -------------------------------------------------------------------
view_mode = st.radio(
    "Select Dashboard View:", 
    ["📋 Activity Log", "📈 Progression", "⚡ Speed vs HR", "🏆 Personal Bests", "🚴 Matched Routes"], 
    horizontal=True
)

if view_mode == "📋 Activity Log":
    st.subheader("📊 Activity Log")
    
    df_display = df_all.copy()
    df_display["Moving Time"] = df_display["moving_time_sec"].apply(format_seconds)

    st.dataframe(
        df_display[["Display Label", "activity_type", "distance_mi", "Moving Time", "avg_speed_mph", "max_speed_mph"]].rename(
            columns={
                "Display Label": "Activity / Date",
                "activity_type": "Type", 
                "distance_mi": "Distance (mi)", 
                "avg_speed_mph": "Avg Speed (mph)", 
                "max_speed_mph": "Max Speed (mph)"
            }
        ),
        column_config={
            "Distance (mi)": st.column_config.NumberColumn(format="%.2f mi"),
            "Avg Speed (mph)": st.column_config.NumberColumn(format="%.1f mph"),
            "Max Speed (mph)": st.column_config.NumberColumn(format="%.1f mph")
        },
        hide_index=True,
        use_container_width=True
    )

elif view_mode == "🚴 Matched Routes":
    st.subheader("🚴 Matched Rides & Repeated Route Intelligence")
    matched_df = df_all[df_all["route_id"] > 0].copy()
    
    if matched_df.empty:
        st.info("No matched routes found. Click 'Re-cluster Matched Routes' in the sidebar or expand your date scope.")
    else:
        route_summary = matched_df.groupby("route_id").agg(
            efforts=("activity_id", "count"),
            avg_dist=("distance_mi", "mean"),
            route_name=("activity_name", "first")
        ).reset_index()

        route_options = {r["route_id"]: f"Route #{r['route_id']}: {r['route_name']} (~{r['avg_dist']:.1f} mi) - {r['efforts']} Efforts" for _, r in route_summary.iterrows()}
        selected_route_id = st.selectbox("Select Matched Route Cluster:", list(route_options.keys()), format_func=lambda x: route_options[x])

        route_efforts = matched_df[matched_df["route_id"] == selected_route_id].sort_values("Activity Date", ascending=True).copy()
        
        all_time_avg_speed = route_efforts["avg_speed_mph"].mean()
        fastest_speed = route_efforts["avg_speed_mph"].max()
        slowest_speed = route_efforts["avg_speed_mph"].min()
        latest_effort = route_efforts.iloc[-1]
        delta_latest = latest_effort["avg_speed_mph"] - all_time_avg_speed

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Efforts", len(route_efforts))
        m2.metric("Fastest Speed", f"{fastest_speed:.1f} mph")
        m3.metric("All-Time Avg Speed", f"{all_time_avg_speed:.1f} mph")
        m4.metric("Latest Pace Delta", f"{delta_latest:+.1f} mph", delta_color="normal")

        fig_matched = go.Figure()
        fig_matched.add_trace(go.Scatter(
            x=route_efforts["Activity Date"],
            y=route_efforts["avg_speed_mph"],
            mode="markers+lines",
            name="Effort Speed",
            marker=dict(size=8, color="#00B4D8"),
            line=dict(width=1, dash="dot"),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Speed: %{y:.1f} mph<extra></extra>"
        ))
        
        if len(route_efforts) >= 3:
            route_efforts["rolling_avg"] = route_efforts["avg_speed_mph"].rolling(3, min_periods=1).mean()
            fig_matched.add_trace(go.Scatter(
                x=route_efforts["Activity Date"],
                y=route_efforts["rolling_avg"],
                mode="lines",
                name="Trending Average",
                line=dict(width=3, color="#70E000")
            ))

        fig_matched.add_hline(y=all_time_avg_speed, line_dash="dash", line_color="gray", annotation_text="All-Time Avg")
        fig_matched.update_layout(
            title="Route Speed Progression Over Time", 
            xaxis_title="Date", 
            yaxis_title="Speed (mph)", 
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E0E1DD")
        )
        st.plotly_chart(fig_matched, use_container_width=True)

        st.markdown("#### Itemized Route Efforts")
        route_efforts["Pace Delta"] = route_efforts["avg_speed_mph"] - all_time_avg_speed
        route_efforts["Moving Time"] = route_efforts["moving_time_sec"].apply(format_seconds)

        st.dataframe(
            route_efforts[["activity_date", "activity_name", "avg_speed_mph", "Pace Delta", "Moving Time", "distance_mi"]].sort_values("activity_date", ascending=False),
            column_config={
                "activity_date": "Date",
                "activity_name": "Activity",
                "avg_speed_mph": st.column_config.NumberColumn("Speed", format="%.1f mph"),
                "Pace Delta": st.column_config.NumberColumn("+/- All-Time Avg", format="%+.1f mph"),
                "distance_mi": st.column_config.NumberColumn("Distance", format="%.2f mi")
            },
            hide_index=True,
            use_container_width=True
        )

elif view_mode == "📈 Progression":
    st.subheader("📈 Over Time Progression")
    fig = px.line(df_all.sort_values("Activity Date"), x="Activity Date", y="avg_speed_mph", color="activity_type", title="Average Speed Over Time")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#E0E1DD"))
    st.plotly_chart(fig, use_container_width=True)

elif view_mode == "⚡ Speed vs HR":
    st.subheader("⚡ Speed vs HR Efficiency")
    eff_df = df_all[df_all["avg_hr"] > 0].copy()
    if not eff_df.empty:
        fig = px.scatter(eff_df, x="avg_hr", y="avg_speed_mph", color="activity_type", size="distance_mi", title="Avg Speed vs Avg HR")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#E0E1DD"))
        st.plotly_chart(fig, use_container_width=True)

elif view_mode == "🏆 Personal Bests":
    st.subheader("🏆 Personal Bests")
    c1, c2, c3 = st.columns(3)
    c1.metric("Longest Distance", f"{df_all['distance_mi'].max():.2f} mi")
    c2.metric("Fastest Avg Speed", f"{df_all['avg_speed_mph'].max():.1f} mph")
    c3.metric("Highest Max Speed", f"{df_all['max_speed_mph'].max():.1f} mph")