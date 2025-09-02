import streamlit as st
import folium
from streamlit_folium import st_folium
import ee
from datetime import datetime
import requests
import openmeteo_requests
import pandas as pd
import altair as alt
from folium.plugins import Geocoder

def calc_irrigation(avg_monthly_et0_df, ndvi_value, irrigation_months, rain_value):
    if avg_monthly_et0_df is None or avg_monthly_et0_df.empty or ndvi_value is None or rain_value is None or not irrigation_months:
        return None
    
    df_result = avg_monthly_et0_df.copy()
    
    # Calculate ETa
    df_result['ETa'] = df_result.apply(
        lambda row: row['ET0'] * (ndvi_value / 0.73) if row['ET0'] is not None else None, axis=1
    )
    
    # Filter for months in irrigation_months
    df_result = df_result[df_result['month'].isin(irrigation_months)].copy()
    
    # Calculate irrigation
    if len(irrigation_months) > 0:
        rain_per_month = (rain_value - 50) / len(irrigation_months)
        df_result['Irrigation'] = df_result.apply(
            lambda row: row['ETa'] - rain_per_month if row['ETa'] is not None else None, axis=1
        )
    else:
        df_result['Irrigation'] = df_result['ETa'] # If no irrigation months, Irrigation is just ETa
        
    # Deficit irrigation logic for July and September
    if 7 in df_result['month'].values:
        july_irrigation_row = df_result[df_result['month'] == 7]
        if not july_irrigation_row.empty:
            july_irrigation = july_irrigation_row['Irrigation'].iloc[0]
            reduction_amount = july_irrigation * 0.20
            
            # Reduce July irrigation
            df_result.loc[df_result['month'] == 7, 'Irrigation'] -= reduction_amount
            
            # Add to September irrigation if it exists
            if 9 in df_result['month'].values:
                df_result.loc[df_result['month'] == 9, 'Irrigation'] += reduction_amount

    # Calculate SW: initial rain minus monthly cumulative irrigation
    # Assuming 'rain_value' is the total initial rain for the period
    df_result['ETa_cum'] = df_result['ETa'].cumsum()
    df_result['SW'] = rain_value + df_result['Irrigation'].cumsum() - df_result['ETa_cum']
        
    return df_result

@st.cache_data(show_spinner=False)
def get_et0(lat, lon):
    today = datetime.today()
    start_date = f"{today.year - 5}-01-01"
    end_date = f"{today.year - 1}-12-31"

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "et0_fao_evapotranspiration",
        "timezone": "auto"
    }

    openmeteo = openmeteo_requests.Client()
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Build dataframe

    daily=response.Daily()
    
    time = pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )

    et0 = response.Daily().Variables(0).ValuesAsNumpy()

    # Build DataFrame
    df = pd.DataFrame({"time": time, "et0": et0})

    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    
    # Step 1: sum ET₀ per (year, month)
    monthly_sums = df.groupby(["year", "month"])["et0"].sum().reset_index()

    # Step 2: average monthly sums across years
    avg_monthly_et0 = monthly_sums.groupby("month")["et0"].mean().reset_index()
    avg_monthly_et0["et0"] = avg_monthly_et0["et0"] * 1.1

    avg_monthly_et0.rename(columns={"et0": "ET0"}, inplace=True)
    
    return avg_monthly_et0

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
def get_rain(lat, lon):
    # Determine start date: Nov 1 of this or last year
    today = datetime.today()
    start_year = today.year - 1 if today.month < 11 else today.year
    start_date = f"{start_year}-11-01"

    # Build API URL
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": today.strftime("%Y-%m-%d"),
        "daily": "rain_sum",
        "timezone": "auto"
    }

    # Fetch and parse data
    openmeteo = openmeteo_requests.Client()
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Extract rain and time values
    time = response.Daily().Time()
    rain = response.Daily().Variables(0).ValuesAsNumpy()

    # Build DataFrame
    df = pd.DataFrame({"time": pd.to_datetime(time), "rain": rain})

    # Return total rainfall
    return round(df["rain"].sum(skipna=True), 1)

st.title("Almond Water Budget")

@st.cache_resource
def initialize_earth_engine():
    ee.Initialize(project="ee-orsperling")

initialize_earth_engine()

# Create a Folium map centered on Fresno
m = folium.Map(location=[32.76558726677407, 35.75073837793036], zoom_start=13, tiles=None)

folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr='Map data © Google',
        name='Google Satellite',
        overlay=False,
        control=True
    ).add_to(m)
Geocoder().add_to(m)
# Add a click event listener to the map
m.add_child(folium.LatLngPopup())

# Display the map in Streamlit and capture events
map_data = st_folium(m, width=700, height=500)

with st.sidebar:
    st.header("Irrigation Month Selection")
    irrigation_month_start, irrigation_month_end = st.slider(
        "Select Irrigation Months (Start-End)",
        min_value=1, max_value=12, value=(3, 11),
        key="irrigation_months_slider"
    )
    irrigation_months = range(irrigation_month_start, irrigation_month_end + 1)

    st.header("Winter Irrigation")
    winter_irrigation = st.slider(
        "Winter Irrigation (mm)",
        min_value=0,
        max_value=800,
        step=10,
        value=0,
        key="winter_irrigation_slider"
    )

if map_data and map_data["last_clicked"] is not None and "lat" in map_data["last_clicked"]:
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    
    # Get NDVI for the clicked point.
    ndvi_value = get_ndvi(lat, lon)
    rain_value = get_rain(lat, lon)
    et0_value = get_et0(lat, lon)
    
    winter_value = rain_value + winter_irrigation

    
    st.write(f"Trees had an NDVI of {ndvi_value:.2f} last May" + " and it rained " + f"{rain_value:.0f} mm since November." )
        
    
    # Calculate initial irrigation recommendation
    initial_display_data = calc_irrigation(et0_value, ndvi_value, irrigation_months, winter_value)
    
    if initial_display_data is not None and not initial_display_data.empty:
        total_irrigation = initial_display_data['Irrigation'].sum()

        with st.sidebar:
            st.header("Water Allocation")
            allocated_amount = st.slider(
                "Total Irrigation (mm)",
                min_value=0,
                max_value=1800,
                step=50,
                value=int(total_irrigation)
            )

        # Adjust irrigation based on user allocation
        final_display_data = initial_display_data.copy()
        if total_irrigation > 0:
            adjustment_factor = allocated_amount / total_irrigation
            final_display_data['Irrigation'] *= adjustment_factor
        
        # Recalculate SW with adjusted irrigation
        final_display_data['SW'] = winter_value + final_display_data['Irrigation'].cumsum() - final_display_data['ETa_cum']

        col1, col2 = st.columns(2)

        with col1:
            display_df = final_display_data[['month', 'ET0', 'Irrigation']].copy()
            display_df['month'] = pd.to_datetime(display_df['month'], format='%m').dt.month_name()
            display_df['ET0'] = (display_df['ET0'] / 5).round() * 5
            display_df['Irrigation'] = (display_df['Irrigation'] / 5).round() * 5
            st.dataframe(display_df.astype({'ET0': 'int', 'Irrigation': 'int'}), hide_index=True)

        with col2:
            chart_data = final_display_data.copy()
            chart_data['month'] = pd.to_datetime(chart_data['month'], format='%m')

            # Column for the green line part (where SW > 0)
            chart_data['ETa_cum_plot'] = chart_data.apply(lambda row: row['ETa_cum'] if row['SW'] > 20 else None, axis=1)
            
            # Column for the red points part (where SW < 0)
            chart_data['ETa_cum_stress'] = chart_data.apply(lambda row: row['ETa_cum'] if row['SW'] < 20 else None, axis=1)

            # Clip SW values at 0 for plotting purposes
            chart_data['SW'] = chart_data['SW'].clip(lower=0)

            base = alt.Chart(chart_data).encode(x=alt.X('month:T', timeUnit='month', title=None))

            # SW area (always blue)
            area = base.mark_area(opacity=0.7, color='#5276A7').encode(
                y=alt.Y('SW', title='Value'),
                tooltip=['month', 'SW']
            )

            # ETa_cum line, always green, and drawn only where SW > 0
            line = base.mark_line(color='green', strokeWidth=3).encode(
                y=alt.Y('ETa_cum_plot:Q', title='ETa Cumulative'),
                tooltip=['month', 'ETa_cum']
            )

            # Red points for where SW < 0
            points = base.mark_point(color='red', filled=True, size=60).encode(
                y=alt.Y('ETa_cum_stress:Q'),
                tooltip=['month', 'ETa_cum']
            )

            # Combine charts
            chart = alt.layer(area, line, points).resolve_scale(y='shared').interactive()

            st.altair_chart(chart, use_container_width=True)
    else:
        st.write("Could not retrieve ET0/Irrigation data for this location and selected months.")
else:
    st.write("Cliack on your orchard location on the map to get started.")
