#Irr app

import os
from pickle import FALSE

# GEE service autentication DO NOT TOUCH ###############################

import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import ee
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

#initialize_ee()

########################################################################


# 🔐 Authenticate Earth Engine
ee.Authenticate()
ee.Initialize()



# Session State Initialization

st.set_page_config(layout="wide")

if 'lat' not in st.session_state:
    st.session_state['lat'] = None
    st.session_state['lon'] = None
    st.session_state['rain'] = None
    st.session_state['ndvi'] = None
    st.session_state['et0'] = None
    st.session_state['irrigation_df'] = None

#NDVI Fetch Function
def get_ndvi(lat, lon):
    poi = ee.Geometry.Point([lon, lat]).buffer(50)
    img = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
        .filterDate(f"{datetime.now().year - 1}-05-01", f"{datetime.now().year - 1}-06-01") \
        .median()

    ndvi = img.normalizedDifference(['B8', 'B4']).reduceRegion(
        reducer=ee.Reducer.median(),
        geometry=poi,
        scale=50
    ).get('nd')

    try:
        return round(ndvi.getInfo(), 2) if ndvi.getInfo() is not None else None
    except Exception as e:
        return None



# Rain Fetch Function
def get_rain(lat, lon):
    today = datetime.now()
    last_november = datetime(today.year - 1 if today.month < 11 else today.year, 11, 1).date()

    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={last_november}&end_date={today.date()}&daily=rain_sum&timezone=auto"

    response = requests.get(url)
    data = response.json()

    if 'daily' not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data['daily'])
    df['time'] = pd.to_datetime(df['time'])
    df.rename(columns={'rain_sum': 'rain'}, inplace=True)

    df_monthly = df.groupby(df['time'].dt.to_period("M")).agg({'rain': 'sum'}).reset_index()
    df_monthly['time'] = df_monthly['time'].dt.to_timestamp()
    df_monthly['month'] = df_monthly['time'].dt.month

    return df_monthly

#ET0 Fetch Function
def get_ET0(lat, lon):
    today = datetime.now().date()
    start_date = datetime(today.year - 5, today.month, 1).date()

    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={today}&daily=et0_fao_evapotranspiration&timezone=auto"

    response = requests.get(url)
    data = response.json()

    if 'daily' not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data['daily'])
    df['time'] = pd.to_datetime(df['time'])
    df.rename(columns={'et0_fao_evapotranspiration': 'ET0'}, inplace=True)

    df_monthly = df.groupby(df['time'].dt.to_period("M")).agg({'ET0': 'sum'}).reset_index()
    df_monthly['time'] = df_monthly['time'].dt.to_timestamp()
    df_monthly['month'] = df_monthly['time'].dt.month
    df_monthly['year'] = df_monthly['time'].dt.year

    df_avg = df_monthly.groupby('month').agg({'ET0': 'mean'}).reset_index()

    return df_avg

#Irrigation Calculation
def calc_irrigation(rain, ndvi, et0, irrigation_months, w_winter):

    if et0 is None or rain is None or ndvi is None or irrigation_months is None or w_winter is None:
        return None  # or pd.DataFrame() or "Missing data"



    adj_wat = st.sidebar.slider("Fix Rain to Field", 0, 40, int(rain['rain'].sum() * 0.03937), step=1, disabled= True)



    df = et0.copy()
    df['NDVI'] = ndvi
    df = pd.merge(df, rain[['month', 'rain']], on='month', how='outer')




    mnts = list(range(irrigation_months[0], irrigation_months[1] + 1))

    df.loc[~df['month'].isin(range(3, 11)), 'ET0'] = 0

    df['rain'] *= 0.03937
    df['ET0'] *= 0.03937 * 0.9

    df['ET1'] = df['ET0'] * df['NDVI'] / 0.7
    df.loc[df['NDVI'] * 1.05 < 0.7, 'ET1'] = df['ET0'] * df['NDVI'] * 1.05 / 0.7

    df['rain1'] = df['rain']




    df.loc[df['month'] == 2, 'rain1'] += w_winter
    rain_sum = df['rain1'].sum()

    if adj_wat != int(rain['rain'].sum() * 0.03937):
        rain_sum = adj_wat





    SWI = ((rain_sum - df.loc[~df['month'].isin(mnts), 'ET1'].sum() - 2)) / len(mnts)


    df.loc[df['month'].isin(mnts), 'irrigation'] = df['ET1'] - SWI
    df['irrigation'] = df['irrigation'].clip(lower=0).fillna(0)


    #note to revisit this
    vst = df.loc[df['month'] == 7, 'irrigation'] * 0.1
    df.loc[df['month'] == 7, 'irrigation'] *= 0.8
    df.loc[df['month'].isin([8, 9]), 'irrigation'] += vst.values[0] if not vst.empty else 0


    df['SW1'] = rain_sum - df['ET1'].cumsum() + df['irrigation'].cumsum()
    df['alert'] = np.where(df['SW1'] < 0, 'drought', 'safe')



    All_water = st.sidebar.slider("Water Allocation", 0, 70, int(df['irrigation'].sum()), step=5, disabled=True)

    if All_water != int(df['irrigation'].sum()):
        delta = All_water - df['irrigation'].sum()
        print(delta)



    return df

#Streamlit UI
# st.title("California Almond Calculator")

st.sidebar.subheader("Farm Data")
w_winter = st.sidebar.slider("Winter Irrigation", 0, 40, 0, step=1)
irrigation_months = st.sidebar.slider("Irrigation Months", 1, 12, (datetime.now().month + 1, 10), step=1)

#Units Toggle
units = st.sidebar.radio("Units (only results)", ["inches", "mm"])
conversion_factor = 1 if units == "inches" else 25.4

col1, col2 = st.columns([6, 4])

with col1:
    st.subheader("Select your farm Location")

    BaseMap = folium.Map(
        location=[35.261723, -119.177502],
        zoom_start=14,
        tiles= "Esri.WorldImagery"
    )

    map_data = st_folium(BaseMap,
        width="100%",
        height=700
    )

with col2:

    st.subheader("Results")


    if map_data and 'last_clicked' in map_data and isinstance(map_data['last_clicked'], dict):
        coords = map_data['last_clicked']
        lat, lon = coords['lat'], coords['lng']

        if (lat != st.session_state['lat']) or (lon != st.session_state['lon']):
            st.session_state['lat'] = lat
            st.session_state['lon'] = lon
            st.session_state['rain'] = get_rain(lat, lon)
            st.session_state['ndvi'] = get_ndvi(lat, lon)
            st.session_state['et0'] = get_ET0(lat, lon)
            folium.Marker([lon, lat], popup='posit').add_to(BaseMap)

    # Move this outside the conditional so it runs on ANY input change
    if 'rain' in st.session_state and 'ndvi' in st.session_state and 'et0' in st.session_state:
        st.session_state['irrigation_df'] = calc_irrigation(
            st.session_state['rain'],
            st.session_state['ndvi'],
            st.session_state['et0'],
            irrigation_months,
            w_winter
        )

    if st.session_state['irrigation_df'] is not None:
        rain = st.session_state['rain']
        ndvi = st.session_state['ndvi']
        df_irrigation = st.session_state['irrigation_df']




        # Plotting
        fig, ax = plt.subplots()
        ax.bar(df_irrigation['month'], df_irrigation['irrigation'] * conversion_factor, color='blue', alpha=0.3, label="Irrigation")
        ax.plot(df_irrigation['month'], df_irrigation['SW1'] * conversion_factor, marker='o', linestyle='-', color='green', label="Soil Water Balance (SW)")

        ax.set_title(f"NDVI: {ndvi:.2f} ; ET0: {df_irrigation['ET0'].sum():.0f} ; Irrigation: {df_irrigation['irrigation'].sum():.0f}")
        ax.set_xlabel("Month")
        ax.set_ylabel(f"Irrigation ({units})")
        ax.legend()
        st.pyplot(fig)

        df_irrigation['week_irrigation'] = df_irrigation['irrigation'] / 4 * conversion_factor
        st.dataframe(df_irrigation[['month', 'ET0', 'week_irrigation']].round(1))
    else:
        st.error("❌ Not enough data found at this location.")


