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


#Irr app

import os

# GEE service autentication DO NOT TOUCH ################################

from google.oauth2 import service_account

# Function to initialize Earth Engine with credentials
def initialize_ee():
    # Get credentials from Streamlit secrets
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/earthengine"]
    )
    # Initialize Earth Engine
    ee.Initialize(credentials)

initialize_ee()

#########################################################################



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


# 🌧 Function to Fetch Weather Data (ET0 & Rain)
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


# 🌍 Interactive Map for Coordinate Selection
def display_map():
    # Map options
    map_types = {
        "Satellite": "Esri.WorldImagery",
        "Street Map": "OpenStreetMap"
    }

    # User selects map type
    selected_map = st.selectbox("Select Map Type:", list(map_types.keys()))

    # Default map center (California)
    map_center = [35.261723, -119.177502]

    # Create Folium map with the selected map type
    m = folium.Map(location=map_center, zoom_start=14, tiles=map_types[selected_map])

    # Display the map in Streamlit
    return st_folium(m)  # , height=700, width=700)


# 📊 Function to Calculate Irrigation
def calc_irrigation():
    df = et0
    df['NDVI'] = ndvi
    df = pd.merge(df, rain[['month', 'rain']], on='month', how='outer')

    mnts = list(range(irrigation_months[0], irrigation_months[1] + 1))

    df.loc[~df['month'].isin(range(3, 11)), 'ET0'] = 0  # Zero ET0 for non-growing months
    df['rain'] *= 0.03937  # Convert rain to inches
    df['ET0'] *= 0.03937 * 0.9  # Convert ET0 to inches with 90% efficiency

    # Adjust ET1 based on NDVI
    df['ET1'] = df['ET0'] * df['NDVI'] / 0.7
    df.loc[df['NDVI'] * 1.05 < 0.7, 'ET1'] = df['ET0'] * df['NDVI'] * 1.05 / 0.7

    # Adjust rainfall
    df['rain1'] = df['rain']  # * m_rain / df['rain'].sum()
    df.loc[df['month'] == 2, 'rain1'] += m_winter  # Winter irrigation

    # # Soil water balance
    SWI = (df['rain1'].sum() - df.loc[~df['month'].isin(mnts), 'ET1'].sum() - 2) / len(mnts)

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


# 🌟 *Streamlit UI*
# st.title("California Almond Calculator")

# 📌 *User Inputs*
st.sidebar.subheader("Farm Data")
# m_rain = st.sidebar.slider("Fix Rain to Field", 0, 40, 10, step=1)
m_winter = st.sidebar.slider("Winter Irrigation", 0, 40, 0, step=1)
irrigation_months = st.sidebar.slider("Irrigation Months", 1, 12, (datetime.now().month + 1, 10), step=1)

# total_irrigation = (1450*ndvi/.7-rain['rain'].sum()+40)*.04
# m_irrigation = st.sidebar.slider("Water Allocation", 0, 70, int(total_irrigation), step=5)


# if "m_rain" not in st.session_state:
#     st.session_state["m_rain"] = 0

# # 📌 Sidebar Slider for Irrigation Months (Reactive to Map Click)
# m_rain = st.sidebar.slider(
#     "Fix Rain", 0, 40, st.session_state["m_rain"], step=1
# )

# Layout: 2 columns (map | output)
col1, col2 = st.columns([3, 5])

with col1:
    # 🗺 *Map Selection*
    map_data = display_map()

with col2:
    if map_data and isinstance(map_data, dict) and 'last_clicked' in map_data and isinstance(map_data['last_clicked'],
                                                                                             dict):

        coords = map_data['last_clicked']
        lat, lon = coords['lat'], coords['lng']

        # st.write(f"📍 Selected Location: *{lat:.2f}, {lon:.2f}*")

        # 📊 Fetch and Process Data
        rain = get_rain(lat, lon)
        ndvi = get_ndvi(lat, lon)
        et0 = get_ET0(lat, lon)

        total_rain = rain['rain'].sum() * 0.04
        m_rain = st.sidebar.slider("Fix Rain to Field", 0, 40, int(total_rain), step=1, disabled=True)

        if lat is not None and lon is not None:

            df_irrigation = calc_irrigation()
            # st.write("NDVI is: ", ndvi, " ; ET0 is", df_irrigation["ET0"].sum().round(0), " ; Irrigation is", df_irrigation["irrigation"].sum().round(0))

            total_irrigation = df_irrigation['irrigation'].sum()
            m_irrigation = st.sidebar.slider("Water Allocation", 0, 70, int(total_irrigation), step=5, disabled=True)

            # 📈 Plot Data
            # st.subheader("Irrigation & Soil Water Balance")
            fig, ax = plt.subplots()
            ax.bar(df_irrigation['month'], df_irrigation['irrigation'], color='blue', alpha=0.5, label="Irrigation")
            ax.plot(df_irrigation['month'], df_irrigation['SW1'], marker='o', linestyle='-', color='red',
                    label="Soil Water Balance (SW)")

            ax.set_title(
                f"NDVI is: {ndvi:.2f} ; ET0 is {df_irrigation['ET0'].sum():.0f} ; Irrigation is {df_irrigation['irrigation'].sum():.0f}")
            ax.set_xlabel("Month")
            ax.set_ylabel("Irrigation (inches)")
            ax.legend()
            st.pyplot(fig)

            df_irrigation['week_irrigation'] = df_irrigation['irrigation'] / 4

            # 📊 Display Table
            # st.subheader("Irrigation Table")
            st.dataframe(df_irrigation[['month', 'ET0', 'week_irrigation']].round(1))

        else:
            st.error("❌ No weather data found at this location.")