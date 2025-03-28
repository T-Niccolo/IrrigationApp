import streamlit as st
import pandas as pd
import folium
import json
import requests
import ee
import numpy as np
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
from datetime import datetime
from shapely.geometry import Point


# Initialize Google Earth Engine
from google.oauth2 import service_account

st.set_page_config(layout='wide')


# Function to initialize Earth Engine with credentials
def initialize_ee():
    # Get credentials from Streamlit secrets
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/earthengine"]
    )
    # Initialize Earth Engine
    ee.Initialize(credentials)

#initialize_ee()


# 🌍 Function to Fetch NDVI from Google Earth Engine
def get_ndvi(lat, lon):
    poi = ee.Geometry.Point([lon, lat])
    img = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
        .filterDate(f"{datetime.now().year - 1}-05-01", f"{datetime.now().year - 1}-06-01") \
        .median()

    ndvi = img.normalizedDifference(['B8', 'B4']).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=poi,
        scale=50
    ).get('nd')

    try:
        return round(ndvi.getInfo(), 2) if ndvi.getInfo() is not None else None
    except Exception as e:
        return None


# 🌧️ Function to Fetch Weather Data (ET0 & Rain)
def get_rain(lat, lon):
    # Calculate last November's date dynamically
    today = datetime.now()
    last_november = datetime(today.year - 1 if today.month < 11 else today.year, 11, 1).date()

    # API Request URL
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={last_november}&end_date={today.date()}&daily=rain_sum&timezone=auto"

    # Fetch data
    response = requests.get(url)
    data = response.json()

    # Check if response contains 'daily' data
    if 'daily' not in data:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(data['daily'])
    df['time'] = pd.to_datetime(df['time'])
    df.rename(columns={'rain_sum': 'rain'}, inplace=True)

    # Aggregate total rain per month since last November
    df_monthly = df.groupby(df['time'].dt.to_period("M")).agg({'rain': 'sum'}).reset_index()
    df_monthly['time'] = df_monthly['time'].dt.to_timestamp()  # Convert Period to Timestamp
    df_monthly['month'] = df_monthly['time'].dt.month

    return df_monthly


def get_rain_era5(lat, lon):
    # Dates: from last Nov 1 to today
    today = datetime.now()
    year = today.year if today.month >= 11 else today.year - 1
    start_date = datetime(year, 11, 1)

    point = ee.Geometry.Point(lon, lat)

    # ERA5-Land: daily total precipitation
    collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
        .filterDate(start_date.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")) \
        .select("total_precipitation_sum")

    # Extract daily values
    def extract(img):
        date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        rain = img.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=1000
        ).get("total_precipitation_sum")
        return ee.Feature(None, {"date": date, "rain": rain})

    fc = ee.FeatureCollection(collection.map(extract))
    dates = fc.aggregate_array("date").getInfo()
    rains = fc.aggregate_array("rain").getInfo()

    # DataFrame: convert and group
    df = pd.DataFrame({"date": pd.to_datetime(dates), "rain": rains})
    df["rain"] = pd.to_numeric(df["rain"], errors="coerce") * 1000  # meters → mm
    df = df.dropna()

    df["month"] = df["date"].dt.to_period("M")
    df_monthly = df.groupby("month")["rain"].sum().reset_index()
    df_monthly["month"] = df_monthly["month"].dt.month

    return df_monthly


def get_ET0(lat, lon):
    # Calculate start date (5 years ago from today)
    today = datetime.now().date()
    start_date = datetime(today.year - 5, today.month, 1).date()  # First day of the month 5 years ago

    # API Request URL
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={today}&daily=et0_fao_evapotranspiration&timezone=auto"

    # Fetch data
    response = requests.get(url)
    data = response.json()

    # Check if response contains 'daily' data
    if 'daily' not in data:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(data['daily'])
    df['time'] = pd.to_datetime(df['time'])
    df.rename(columns={'et0_fao_evapotranspiration': 'ET0'}, inplace=True)

    # Calculate monthly average ET0 over the last 5 years
    df_monthly = df.groupby(df['time'].dt.to_period("M")).agg({'ET0': 'sum'}).reset_index()
    df_monthly['time'] = df_monthly['time'].dt.to_timestamp()  # Convert Period to Timestamp
    df_monthly['month'] = df_monthly['time'].dt.month
    df_monthly['year'] = df_monthly['time'].dt.year

    # Calculate 5-year average ET0 for each month
    df_avg = df_monthly.groupby('month').agg({'ET0': 'mean'}).reset_index()

    return df_avg


def get_et0_gridmet(lat, lon):
    # 5 full years
    today = datetime.now()
    start_date = datetime(today.year - 5, 1, 1)
    end_date = datetime(today.year - 1, 12, 31)

    point = ee.Geometry.Point(lon, lat)

    # Load GRIDMET ET₀ (etr in mm/day)
    collection = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET") \
        .filterDate(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")) \
        .select("eto")

    # Extract daily ET₀ and date
    def extract(img):
        date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        et0 = img.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=4000
        ).get("eto")
        return ee.Feature(None, {"date": date, "et0": et0})

    fc = ee.FeatureCollection(collection.map(extract))
    dates = fc.aggregate_array("date").getInfo()
    et0_vals = fc.aggregate_array("et0").getInfo()

    # Convert to DataFrame
    df = pd.DataFrame({"date": pd.to_datetime(dates), "et0": et0_vals})
    df["et0"] = pd.to_numeric(df["et0"], errors="coerce")
    df = df.dropna()

    # Group by month and year
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df_monthly = df.groupby(["year", "month"])["et0"].sum().reset_index()

    # Now get average ET₀ per calendar month across years
    avg_monthly_et0 = df_monthly.groupby("month")["et0"].mean().reset_index()
    avg_monthly_et0.rename(columns={"et0": "ET0"}, inplace=True)

    return avg_monthly_et0


# 🌍 Interactive Map for Coordinate Selection
def display_map():
    # Center and zoom
    map_center = [35.24736288352025, -119.18877345475644]
    zoom = 14

    # Create base map with no tiles
    m = folium.Map(location=map_center, zoom_start=zoom, tiles=None)

    # Satellite base layer
    folium.TileLayer(
        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite",
        overlay=False,
        control=False
    ).add_to(m)

    # Transparent place labels (cities, landmarks)
    folium.TileLayer(
        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Boundaries & Labels",
        name="Place Labels",
        overlay=True,
        control=False
    ).add_to(m)

    # Transparent road overlay (includes road numbers!)
    folium.TileLayer(
        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Transportation",
        name="Roads",
        overlay=True,
        control=False
    ).add_to(m)

    return st_folium(m, height=600, width=900)


# 📊 Function to Calculate Irrigation
def calc_irrigation(ndvi, rain, et0, m_winter, irrigation_months):
    df = et0
    df['NDVI'] = ndvi
    df = pd.merge(df, rain[['month', 'rain']], on='month', how='outer')

    mnts = list(range(irrigation_months[0], irrigation_months[1] + 1))

    df.loc[~df['month'].isin(range(3, 11)), 'ET0'] = 0  # Zero ET0 for non-growing months
    df['rain'] *= conversion_factor  # Convert rain to inches
    df['ET0'] *= conversion_factor * 0.9  # Convert ET0 to inches with 90% efficiency

    # Adjust ET1 based on NDVI
    df['ET1'] = df['ET0'] * df['NDVI'] / 0.7
    df.loc[df['NDVI'] * 1.05 < 0.7, 'ET1'] = df['ET0'] * df['NDVI'] * 1.05 / 0.7

    # Adjust rainfall
    df['rain1'] = df['rain']  # * m_rain / df['rain'].sum()
    df.loc[df['month'] == 2, 'rain1'] += m_winter  # Winter irrigation

    # # Soil water balance
    SWI = (df['rain1'].sum() - df.loc[~df['month'].isin(mnts), 'ET1'].sum() - 50 * conversion_factor) / len(mnts)

    df.loc[df['month'].isin(mnts), 'irrigation'] = df['ET1'] - SWI
    df['irrigation'] = df['irrigation'].clip(lower=0)
    df['irrigation'] = df['irrigation'].fillna(0)

    vst = df.loc[df['month'] == 7, 'irrigation'] * 0.1
    df.loc[df['month'] == 7, 'irrigation'] *= 0.8
    df.loc[df['month'].isin([8, 9]), 'irrigation'] += vst.values[0] if not vst.empty else 0

    # df['irrigation'] *= m_irrigation / df['irrigation'].sum()
    df['SW1'] = df['rain1'].sum() - df['ET1'].cumsum() + df['irrigation'].cumsum()
    df['alert'] = np.where(df['SW1'] < 0, 'drought', 'safe')

    return df  # [['time', 'ET0', 'ET1', 'rain', 'rain1', 'irrigation', 'SW0', 'SW1', 'alert']]


# 🌟 **Streamlit UI**
# st.title("California Almond Calculator")
# 📌 **User Inputs**
# 🌍 Unit system selection
unit_system = st.sidebar.radio("Select Units", ["Imperial (inches)", "Metric (mm)"])

unit_label = "inches" if "Imperial" in unit_system else "mm"
conversion_factor = 0.03937 if "Imperial" in unit_system else 1

st.sidebar.subheader("Farm Data")
# m_winter = st.sidebar.slider("Winter Irrigation", 0, 40, 0, step=1)
# irrigation_months = st.sidebar.slider("Irrigation Months", 1, 12, (datetime.now().month + 1, 10), step=1)

# Layout: 2 columns (map | output)
col1, col2 = st.columns([3, 5])

with col1:
    # 🗺️ **Map Selection**
    map_data = display_map()

with col2:
    # --- Sliders (trigger irrigation calc only)
    m_winter = st.sidebar.slider("Winter Irrigation", 0, int(round(700 * conversion_factor)), 0,
                                 step=int(round(20 * conversion_factor)))
    irrigation_months = st.sidebar.slider("Irrigation Months", 1, 12, (datetime.now().month + 1, 10), step=1)

    # --- Handle map click
    if map_data and isinstance(map_data, dict) and "last_clicked" in map_data:
        coords = map_data["last_clicked"]

        # ✅ Only proceed if coords is valid
        if coords and "lat" in coords and "lng" in coords:
            lat, lon = coords["lat"], coords["lng"]
            location = (round(lat, 5), round(lon, 5))

            # Check if location changed
            location_changed = st.session_state.get("last_location") != location

            if location_changed:
                st.session_state["last_location"] = location
                st.session_state["rain"] = get_rain_era5(lat, lon)
                st.session_state["ndvi"] = get_ndvi(lat, lon)
                st.session_state["et0"] = get_et0_gridmet(lat, lon)

            # Retrieve stored values
            rain = st.session_state.get("rain")
            ndvi = st.session_state.get("ndvi")
            et0 = st.session_state.get("et0")

            if rain is not None and ndvi is not None and et0 is not None:
                total_rain = rain['rain'].sum() * conversion_factor
                m_rain = st.sidebar.slider("Fix Rain to Field", 0, int(round(1000 * conversion_factor)),
                                           int(total_rain), step=1, disabled=True)

                # 🔄 Always recalculate irrigation when sliders or location change
                df_irrigation = calc_irrigation(ndvi, rain, et0, m_winter, irrigation_months)

                total_irrigation = df_irrigation['irrigation'].sum()
                m_irrigation = st.sidebar.slider("Water Allocation", 0, int(round(1500 * conversion_factor)),
                                                 int(total_irrigation), step=5, disabled=True)

                # 📈 Plot
                fig, ax = plt.subplots()
                ax.bar(df_irrigation['month'], df_irrigation['irrigation'], color='blue', alpha=0.5, label="Irrigation")
                ax.plot(df_irrigation['month'], df_irrigation['SW1'], marker='o', linestyle='-', color='red',
                        label="Soil Water Balance (SW)")
                ax.set_title(
                    f"NDVI: {ndvi:.2f} | ET₀: {df_irrigation['ET0'].sum():.0f} | Irrigation: {total_irrigation:.0f}")
                ax.set_xlabel("Month")
                ax.set_ylabel("Irrigation")
                ax.legend()
                st.pyplot(fig)

                # 📊 Table
                df_irrigation['week_irrigation'] = df_irrigation['irrigation'] / 4

                # Filter by selected irrigation months
                start_month, end_month = irrigation_months
                filtered_df = df_irrigation[df_irrigation['month'].between(start_month, end_month)]

                # Show only monthly ET₀ and irrigation totals
                filtered_df.index = [''] * len(filtered_df)
                st.dataframe(filtered_df[['month', 'ET0', 'week_irrigation']].round(1))

            else:
                st.error("❌ No weather data found for this location.")
        else:
            st.info("🖱️ Click a location on the map to begin.")
    else:
        st.info("🖱️ Click a location on the map to begin.")


