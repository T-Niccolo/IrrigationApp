import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import folium
import ee
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
from shapely.geometry import Point
from google.oauth2 import service_account
from PIL import Image

st.set_page_config(layout='wide')


ee.Authenticate()
ee.Initialize()


# 🌍 Function to Fetch NDVI from Google Earth Engine
@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
def get_rain_era5(lat, lon):

    # Define date range
    today = datetime.now()
    start_year = today.year if today.month >= 11 else today.year - 1
    start = f"{start_year}-11-01"
    end = today.strftime("%Y-%m-%d")

    # Define location
    point = ee.Geometry.Point(lon, lat)

    # Get total precipitation image
    rain_sum = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
        .filterDate(start, end) \
        .select("total_precipitation_sum") \
        .sum()

    # Reduce to value at point
    try:
        rain_mm = rain_sum.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=1000
        ).get("total_precipitation_sum").getInfo()

        return rain_mm * 1000  # Convert meters to mm
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def get_et0_gridmet(lat, lon):
    
    today = datetime.now()
    start_date = datetime(today.year - 5, 1, 1)
    end_date = datetime(today.year - 1, 12, 31)

    point = ee.Geometry.Point(lon, lat)

    # Start and end as ee.Date
    start = ee.Date(start_date.strftime("%Y-%m-%d"))
    end = ee.Date(end_date.strftime("%Y-%m-%d"))

    # Create monthly steps (5 full years = 60 months)
    month_count = end.difference(start, 'month')
    months = ee.List.sequence(0, month_count.subtract(1))

    def monthly_sum(n):
        start_month = start.advance(n, 'month')
        end_month = start_month.advance(1, 'month')
        monthly_img = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET") \
            .filterDate(start_month, end_month) \
            .select("eto") \
            .sum()

        value = monthly_img.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=4000
        ).get("eto")

        return ee.Feature(None, {
            "month": start_month.format("M"),
            "year": start_month.format("Y"),
            "et0": value
        })

    # Map and fetch features
    fc = ee.FeatureCollection(months.map(monthly_sum))

    try:
        features = fc.getInfo()["features"]
        data = [{"month": int(f["properties"]["month"]),
                 "year": int(f["properties"]["year"]),
                 "et0": f["properties"]["et0"]} for f in features]
    except Exception:
        return None

    df = pd.DataFrame(data)
    df["et0"] = pd.to_numeric(df["et0"], errors="coerce")
    df = df.dropna()

    if df.empty:
        return None

    # Average ET0 per calendar month over the years
    avg_monthly_et0 = df.groupby("month")["et0"].mean().reset_index()
    avg_monthly_et0.rename(columns={"et0": "ET0"}, inplace=True)

    return avg_monthly_et0

#DEFAULT_CENTER = [35.26, -119.15]
#DEFAULT_ZOOM = 13

# 🌍 Interactive Map for Coordinate Selection
def display_map():
    
    # Center and zoom
    map_center = [35.26, -119.15]
    zoom = 13


    # Get session state to store the clicked location
    #if "clicked_location" not in st.session_state:
    #    st.session_state.clicked_location = None  # No click yet
    #    print("noclick")

    #test = st.session_state.get("clicked_location")
    #print("Clicked location:", test)

    # Set initial center
    #map_center = st.session_state.clicked_location if st.session_state.clicked_location else DEFAULT_CENTER


    # Create map
    m = folium.Map(location=map_center, zoom_start=zoom, tiles=None)

    # Add marker only if user clicked
    #if st.session_state.clicked_location:
    #    print("clicked")
    #    folium.Marker(
    #        st.session_state.clicked_location,
    #        popup="Clicked Location",
    #        icon=folium.Icon(color="red")
    #    ).add_to(m)

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

    return st_folium(m, height=600, width=900, use_container_width=True)


# 📊 Function to Calculate Irrigation
def calc_irrigation(ndvi, rain, et0, m_winter, irrigation_months, irrigation_factor):

    df = et0.copy()

    NDVI = ndvi
    rain1 = rain * conversion_factor + m_winter

    if NDVI < 0.67: NDVI *= 1.05

    mnts = list(range(irrigation_months[0], irrigation_months[1] + 1))

    df.loc[~df['month'].isin(range(3, 11)), 'ET0'] = 0  # Zero ET0 for non-growing months
    df['ET0'] *= conversion_factor * 0.8  # Convert ET0 to inches with 90% efficiency

    # Adjust ET1 based on NDVI
    df['ET1'] = df['ET0'] * NDVI / 0.7

    # # Soil water balance
    SWI = (rain1 - df.loc[~df['month'].isin(mnts), 'ET1'].sum() - 50 * conversion_factor) / len(mnts)

    df.loc[df['month'].isin(mnts), 'irrigation'] = df['ET1'] - SWI
    df['irrigation'] = df['irrigation'].clip(lower=0)
    df['irrigation'] = df['irrigation'].fillna(0)
    df["irrigation"] *= irrigation_factor

    vst = df.loc[df['month'] == 7, 'irrigation'] * 0.1
    df.loc[df['month'] == 7, 'irrigation'] *= 0.8
    df.loc[df['month'].isin([8, 9]), 'irrigation'] += vst.values[0] if not vst.empty else 0

    df['SW1'] = (rain1 - df['ET1'].cumsum() + df['irrigation'].cumsum()).clip(lower=0)

    df['alert'] = np.where(df['SW1'] == 0, 'drought', 'safe')

    return df


# 🌟 **Streamlit UI**
st.title("ALMOND - irrigation Monthly Annual Planner")

# 📌 **User Inputs**
# 🌍 Unit system selection

# st.sidebar.caption('This is a research report. For further information contact **Or Sperling** (orsp@volcani.agri.gov.il; ARO-Volcani), **Maciej Zwieniecki** (mzwienie@ucdavis.edu; UC Davis), or **Niccolo Tricerri** (niccolo.tricerri@unito.it; University of Turin).')
st.sidebar.image("img/Logo.png", caption= "**i**rrigation - **M**onthly **A**nnual **P**lanner")

st.sidebar.header("Farm Data")
unit_system = st.sidebar.radio("Select Units", ["Imperial (inches)", "Metric (mm)"], help='What measures do you use?')

unit_label = "inches" if "Imperial" in unit_system else "mm"
conversion_factor = 0.03937 if "Imperial" in unit_system else 1


# Layout: 2 columns (map | output)
col2, col1 = st.columns([6, 4])

if "map_clicked" not in st.session_state:
    st.session_state.map_clicked = False

with col1:
    st.header("Select Location")
    
    # 🗺️ **Map Selection**
    map_data = display_map()


    if isinstance(map_data, dict) and (coords := map_data.get("last_clicked")) and {"lat", "lng"} <= coords.keys():
        st.info("Report updated. Change you parameters or select any new location.")

    else:
        st.info("🖱️ Click a location on the map to begin.")


with col2:
    st.write("This is a research report. For further information contact **Or Sperling** (orsp@volcani.agri.gov.il; ARO-Volcani), **Maciej Zwieniecki** (mzwienie@ucdavis.edu; UC Davis), **Zac Ellis** (zellis@ucdavis.edu; UC Davis), or **Niccolo' Tricerri** (niccolo.tricerri@unito.it; University of Turin - IUSS Pavia).")

    # --- Sliders (trigger irrigation calc only)
    m_winter = st.sidebar.slider(f"Winter Irrigation ({unit_label})", 0, int(round(700 * conversion_factor)), 0,
                                 step=int(round(20 * conversion_factor)), help="Did you irrigate in winter? If yes, how much?")
    irrigation_months = st.sidebar.slider("Irrigation Months", 1, 12, (datetime.now().month + 1, 10), step=1, help="During which months will you irrigate?")

    irrigation_rate = st.sidebar.slider(f'Irrigation Rate ({unit_label}/hour)', float(.3 * conversion_factor), float(2.8 * conversion_factor), float(1 * conversion_factor), float(.1 * conversion_factor), help="What is your hourly flow rate?")

    # --- Handle map click
    if map_data and isinstance(map_data, dict) and "last_clicked" in map_data:
        coords = map_data["last_clicked"]

        # ✅ Only proceed if coords is valid
        if coords and "lat" in coords and "lng" in coords:

            lat, lon = coords["lat"], coords["lng"]

            location = (lat, lon)

            # Check if location changed
            now = time.time()
            last_loc = st.session_state.get("last_location")
            last_time = st.session_state.get("last_location_time", 0)

            # location_changed = (last_loc != location) and (now - last_time > 5)

            if location != last_loc or (now - last_time > 5):
                # Update session state with the new location and timestamp
                st.session_state["last_location"] = location
                st.session_state["last_location_time"] = now

                # Fetch and store weather data
                st.session_state["et0"] = get_et0_gridmet(lat, lon)
                st.session_state["rain"] = get_rain_era5(lat, lon)
                st.session_state["ndvi"] = get_ndvi(lat, lon)

            # Retrieve stored values
            rain = st.session_state.get("rain")
            ndvi = st.session_state.get("ndvi")
            et0 = st.session_state.get("et0")

            if rain is not None and ndvi is not None and et0 is not None:
                total_rain = rain * conversion_factor
                m_rain = st.sidebar.slider(f"Fix Rain to Field ({unit_label})", 0, int(round(1000 * conversion_factor)),
                                           int(total_rain), step=1, disabled=True)

                # 🔄 Always recalculate irrigation when sliders or location change
                df_irrigation = calc_irrigation(ndvi, rain, et0, m_winter, irrigation_months, 1)

                total_irrigation = df_irrigation['irrigation'].sum()
                m_irrigation = st.sidebar.slider(f"Water Allocation ({unit_label})", 0, int(round(1500 * conversion_factor)),
                                                 int(total_irrigation), step=int(round(20 * conversion_factor)),
                                                 help="Here's the recommended irrigation. Are you constrained by water availability, or considering extra irrigation for salinity management?")

                irrigation_factor = m_irrigation / total_irrigation

                # ✅ Adjust ET0 in the table
                df_irrigation = calc_irrigation(ndvi, rain, et0, m_winter, irrigation_months, irrigation_factor)
                total_irrigation = df_irrigation['irrigation'].sum()

                # 📈 Plot
                fig, ax = plt.subplots()
                ax.bar(df_irrigation['month'], df_irrigation['irrigation'], color='#3897c5', alpha=1, label="Irrigation")
                ax.plot(df_irrigation['month'], df_irrigation['SW1'], marker='o', linestyle='-', color='#74ac72',
                        label="Soil Water")
                # Red overlay where SW1 < 0
                df_below_zero = df_irrigation[df_irrigation['SW1'] <= 0]
                if not df_below_zero.empty:
                    ax.plot(df_below_zero['month'], df_below_zero['SW1'], marker='o', linestyle='None', color='#FF4B4B',
                            label="Drought")

                ax.set_ylim(bottom=-3.7 * conversion_factor)

                ax.set_title(
                    f"NDVI: {ndvi:.2f} | ET₀: {df_irrigation['ET0'].sum():.0f} {unit_label} | Irrigation: {total_irrigation:.0f} {unit_label}")
                ax.set_xlabel("Month")
                ax.set_ylabel(f"Water amount ({unit_label})")
                ax.legend()
                st.pyplot(fig)

                # 📊 Table
                st.subheader('Weekly Irrigation Updates:')
                df_irrigation['week_irrigation_volume'] = df_irrigation['irrigation'] / 4

                # Filter by selected irrigation months
                start_month, end_month = irrigation_months
                filtered_df = df_irrigation[df_irrigation['month'].between(start_month, end_month)]

                # Show only monthly ET₀ and irrigation totals
                filtered_df.index = [''] * len(filtered_df)

                filtered_df['week_irrigation_hours'] = ((filtered_df['week_irrigation_volume'] / irrigation_rate) / .5).round() * .5

                filtered_df['month'] = pd.to_datetime(filtered_df['month'], format='%m').dt.month_name()
                st.dataframe(
                    filtered_df[['month', 'ET0', 'week_irrigation_volume', 'week_irrigation_hours', 'alert']]
                    .rename(columns={
                        'month': 'Month',
                        'ET0': f'ET₀ ({unit_label})',
                        'week_irrigation_volume': f'Irrigation Volume ({unit_label})',
                        'week_irrigation_hours': f'Irrigation time (hours)',
                        'alert': 'Alert'
                    }).round(1),
                    hide_index=True
                )

            else:
                st.error("❌ No weather data found for this location.")
        else:
            image = Image.open("img/ExampleGraph.png")  # Assuming "images" folder in your repo
            st.image(image, caption="Example image of the graphical output", use_container_width=True)

    else:
        st.info("🖱️ Click a location on the map to begin.")