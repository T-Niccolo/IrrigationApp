import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import folium
import ee
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
from google.oauth2 import service_account
from fpdf import FPDF
import tempfile
import os
from PIL import Image
from io import BytesIO
from staticmap import StaticMap, IconMarker

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

initialize_ee()

#ee.Initialize()
#ee.Authenticate()


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
    rain_sum = ee.ImageCollection("OREGONSTATE/PRISM/AN81d") \
        .filterDate(start, end) \
        .select("ppt") \
        .sum()

    # Reduce to value at point
    try:
        rain_mm = rain_sum.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=4638.3
        ).get("ppt").getInfo()

        return rain_mm  # Convert meters to mm
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


# DEFAULT_CENTER = [35.26, -119.15]
# DEFAULT_ZOOM = 13

# 🌍 Interactive Map for Coordinate Selection
def display_map():
    # Center and zoom
    map_center = [35.26, -119.15]
    zoom = 12

    # Create map
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


def save_map_as_image_static(lat, lon, zoom=15, size=(600, 450), marker_path='img/Marker.png', marker_width=50):
    # Esri satellite tile provider
    esri_satellite_url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

    # Create static map
    m = StaticMap(size[0], size[1], url_template=esri_satellite_url)

    # Open and resize marker image by width (keeping aspect ratio)
    with Image.open(marker_path) as img:
        # Get original aspect ratio
        aspect_ratio = img.height / img.width
        new_height = int(marker_width * aspect_ratio)
        resized_img = img.resize((marker_width, new_height), Image.Resampling.LANCZOS)

        buffer = BytesIO()
        resized_img.save(buffer, format='PNG')
        buffer.seek(0)

    # Create and add marker
    marker = IconMarker((lon, lat), buffer, offset_x=marker_width // 2, offset_y=new_height)
    m.add_marker(marker)

    # Render and save
    image = m.render(zoom=zoom)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        image_path = tmp_file.name
        image.save(image_path)

    return image_path


# 🌟 **Streamlit UI**
st.markdown("<h1 style='text-align: center;'>ALMOND - irrigation Monthly Annual Planner</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 20px'>This is a research report founded by <a href=\"https://www.bard-isus.org/\"> <strong>BARD</strong></a>. </p>",
    unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>For further information contact: <a href=\"mailto:orsp@volcani.agri.gov.il\"> <strong>Or Sperling</strong></a> (ARO-Volcani), <a href=\"mailto:mzwienie@ucdavis.edu\"> <strong>Maciej Zwieniecki</strong></a> (UC Davis), <a href=\"mailto:zellis@ucdavis.edu\"> <strong>Zac Ellis</strong></a> (UC Davis), <a href=\"mailto:niccolo.tricerri@unito.it\"> <strong>Niccolò Tricerri</strong></a> (UNITO - IUSS Pavia)  </p>",
    unsafe_allow_html=True)

# 📌 **User Inputs**
# 🌍 Unit system selection

# st.sidebar.caption('This is a research report. For further information contact **Or Sperling** (orsp@volcani.agri.gov.il; ARO-Volcani), **Maciej Zwieniecki** (mzwienie@ucdavis.edu; UC Davis), or **Niccolo Tricerri** (niccolo.tricerri@unito.it; University of Turin).')
st.sidebar.image("img/Logo.png", caption="**i**rrigation - **M**onthly **A**nnual **P**lanner")

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
        st.info(
            "Report updated. Change you parameters or select any new location. You can generate and download a pdf report")

    else:
        st.info("🖱️ Click a location on the map to begin.")

with col2:
    # --- Sliders (trigger irrigation calc only)
    m_winter = st.sidebar.slider(f"Winter Irrigation ({unit_label})", 0, int(round(700 * conversion_factor)), 0,
                                 step=int(round(20 * conversion_factor)),
                                 help="Did you irrigate in winter? If yes, how much?")
    irrigation_months = st.sidebar.slider("Irrigation Months", 1, 12, (datetime.now().month + 1, 10), step=1,
                                          help="During which months will you irrigate?")

    irrigation_rate = st.sidebar.slider(f'Irrigation Rate ({unit_label}/hour)', float(.3 * conversion_factor),
                                        float(2.8 * conversion_factor), float(1 * conversion_factor),
                                        float(.1 * conversion_factor), help="What is your hourly flow rate?")

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
                                           int(total_rain), step=1, disabled=False,
                                           help="Do you know a better value? Do you think less water was retained in the soil?")

                # 🔄 Always recalculate irrigation when sliders or location change
                df_irrigation = calc_irrigation(ndvi, m_rain / conversion_factor, et0, m_winter, irrigation_months, 1)

                total_irrigation = df_irrigation['irrigation'].sum()
                m_irrigation = st.sidebar.slider(f"Water Allocation ({unit_label})", 0,
                                                 int(round(1500 * conversion_factor)),
                                                 int(total_irrigation), step=int(round(20 * conversion_factor)),
                                                 help="Here's the recommended irrigation. Are you constrained by water availability, or considering extra irrigation for salinity management?")

                irrigation_factor = m_irrigation / total_irrigation

                # ✅ Adjust ET0 in the table
                df_irrigation = calc_irrigation(ndvi, m_rain / conversion_factor, et0, m_winter, irrigation_months,
                                                irrigation_factor)
                total_irrigation = df_irrigation['irrigation'].sum()

                st.markdown("""
                <style>
                .centered-stats {
                    text-align: center;
                    font-size: 27px;
                    font-weight: bold;
                    margin-top: 15px;
                }

                .tooltip-icon {
                  display: inline-block;
                  width: 17px;
                  height: 17px;
                  background-color: #3498db;
                  color: white;
                  border-radius: 50%;
                  text-align: center;
                  font-size: 12px;
                  line-height: 16px;
                  margin-left: 3px;
                  cursor: help;
                  vertical-align: middle;
                }
                .tooltip-icon::after {
                  content: "i";
                }
                </style>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="centered-stats">
                  <span title="Potential Normalized Difference Vegetation Index — shows vegetation health."> <span class="tooltip-icon"></span> NDVI</span>: {ndvi:.2f} | 
                  <span title="Potential Total Evapotranspiration for the season"> <span class="tooltip-icon"></span> ET₀</span>: {df_irrigation['ET0'].sum():.0f} {unit_label} | 
                  <span title="Total amount of water suggested for the season."> <span class="tooltip-icon"></span> Irrigation</span>: {total_irrigation:.0f} {unit_label}
                </div>
                """, unsafe_allow_html=True)

                # 📈 Plot
                fig, ax = plt.subplots()
                ax.bar(df_irrigation['month'], df_irrigation['irrigation'], color='#3897c5', alpha=1,
                       label="Irrigation")
                ax.plot(df_irrigation['month'], df_irrigation['SW1'], marker='o', linestyle='-', color='#74ac72',
                        label="Soil Water")
                # Red overlay where SW1 < 0
                df_below_zero = df_irrigation[df_irrigation['SW1'] <= 0]
                if not df_below_zero.empty:
                    ax.plot(df_below_zero['month'], df_below_zero['SW1'], marker='o', linestyle='None', color='#FF4B4B',
                            label="Drought")

                ax.set_ylim(bottom=-3.7 * conversion_factor)

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

                filtered_df['week_irrigation_hours'] = ((filtered_df[
                                                             'week_irrigation_volume'] / irrigation_rate) / .5).round() * .5

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


                # Function to create a PDF report
                def create_pdf_report(fig, df_irrigation, ndvi, total_irrigation, unit_label, NotesText, lat, lon):

                    pdf = FPDF()
                    pdf.add_page()

                    pdf.set_font("Arial", 'B', 20)
                    pdf.cell(0, 10, "ALMOND - iMAP", ln=True, align="L")
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(0, 9, "irrigation Monthly Annual Planner Report for ALMOND orchards", ln=True, align="L")
                    pdf.image("img/Logo.png", x=137, y=10, w=80)

                    pdf.ln(2)
                    pdf.set_font("Arial", size=13)
                    # pdf.cell(0, 8, f"NDVI: {ndvi}", ln=True)

                    month_names = pd.to_datetime(irrigation_months, format='%m').month_name()
                    month_names_str = ' - '.join(month_names)
                    pdf.cell(0, 8, f"Selected irrigation period: {month_names_str}", ln=True)

                    # pdf.cell(0, 8, f"Total Irrigation: {total_irrigation:.2f} {unit_label}", ln=True)
                    pdf.cell(0, 8, f"Irrigation rate: {irrigation_rate:.2f} {unit_label}/hour", ln=True)

                    pdf.set_font("Arial", style='I', size=13)
                    half_width = (pdf.w - 2 * pdf.l_margin) - 50

                    pdf.multi_cell(half_width, 8, f"Notes: {NotesText}")

                    pdf.ln(2)
                    pdf.cell(0, 8, "Weekly plan:", ln=True)

                    df_irrigation['week_irrigation_volume'] = df_irrigation['irrigation'] / 4

                    # Filter by selected irrigation months
                    start_month, end_month = irrigation_months
                    filtered_df = df_irrigation[df_irrigation['month'].between(start_month, end_month)]

                    # Show only monthly ET₀ and irrigation totals
                    filtered_df.index = [''] * len(filtered_df)

                    filtered_df['week_irrigation_hours'] = ((filtered_df[
                                                                 'week_irrigation_volume'] / irrigation_rate) / .5).round() * .5

                    filtered_df['month'] = pd.to_datetime(filtered_df['month'], format='%m').dt.month_name()

                    selected_columns_df = (
                        filtered_df[['month', 'ET0', 'week_irrigation_volume', 'week_irrigation_hours', 'alert']]
                        .rename(columns={
                            'month': 'Month',
                            'ET0': f'ET0 ({unit_label})',
                            'week_irrigation_volume': f'Irrigation Volume ({unit_label})',
                            'week_irrigation_hours': f'Irrigation time (hours)',
                            'alert': 'Alert'
                        })
                        .round(1))

                    # Add table headers
                    headers = selected_columns_df.columns.tolist()

                    # Set custom column widths for each column
                    column_widths = [25, 35, 56, 48, 28]  # Define a width for each column

                    # Set font for headers
                    pdf.set_font("Arial", 'B', 12)
                    for i, header in enumerate(headers):
                        pdf.cell(column_widths[i], 10, header, border=1, align='C')  # Use the respective column width
                    pdf.ln()  # Line break after headers

                    # Set font for data rows
                    pdf.set_font("Arial", '', 12)

                    # Iterate over each row of the filtered DataFrame
                    for index, row in selected_columns_df.iterrows():
                        for i, value in enumerate(row):
                            pdf.cell(column_widths[i], 10, str(value), border=1,
                                     align='C')  # Use respective width for each column
                        pdf.ln()  # Line break after each row

                    pdf.ln(3)

                    pdf.set_font("Arial", size=13)
                    pdf.cell(0, 8,
                             f"data for Location (lat: {round(lat, 3)} lon: {round(lon, 3)}) - NDVI: {ndvi} - Total Irrigation: {total_irrigation:.2f} {unit_label}",
                             ln=True, align='C')
                    pdf.cell(0, 8, f"", ln=True)

                    pdf.ln(3)

                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                        fig.savefig(tmpfile.name, format="PNG", bbox_inches='tight')
                        tmpfile.seek(0)
                        image_path = tmpfile.name

                    image_width = 100
                    image_height = 95

                    bottom_margin = pdf.b_margin
                    y_position = pdf.h - bottom_margin - image_height

                    pdf.image(image_path, x=pdf.l_margin, y=y_position, w=image_width)
                    os.remove(image_path)

                    image_path = save_map_as_image_static(lat, lon)
                    image_width = 87
                    image_height = 93
                    right_margin = pdf.r_margin
                    bottom_margin = pdf.b_margin

                    # Calculate X and Y for bottom-right positioning
                    x_position = pdf.w - right_margin - image_width
                    y_position = pdf.h - bottom_margin - image_height

                    pdf.image(image_path, x=x_position, y=y_position, w=image_width)
                    os.remove(image_path)

                    current_date = datetime.now().strftime("%B %d, %Y")
                    pdf.set_y(-33)
                    pdf.set_font("Arial", size=11)
                    pdf.cell(0, 10, current_date, 0, 0, 'L')
                    pdf.cell(0, 10, "Report based on the work of: Or, Maciej, Zac and Niccolò.", 0, 0, 'R')

                    pdf_bytes = pdf.output(dest="S").encode("latin1")
                    return pdf_bytes


                NotesText = st.text_input("Add your notes for the report:")

                if all(key in st.session_state for key in ["ndvi", "et0"]):
                    if "pdf_ready" not in st.session_state:
                        st.session_state["pdf_ready"] = False
                    if "pdf_downloaded" not in st.session_state:
                        st.session_state["pdf_downloaded"] = False

                    if st.button("⚙️ Generate PDF Report"):
                        with st.spinner("🛠️ Generating PDF report..."):
                            pdf_data = create_pdf_report(
                                fig, df_irrigation, st.session_state["ndvi"],
                                total_irrigation, unit_label, NotesText, lat, lon
                            )
                            st.session_state["pdf_data"] = pdf_data
                            st.session_state["pdf_ready"] = True
                            st.session_state["pdf_downloaded"] = False
                            st.success("✅ PDF report generated!")

                    # Show download button only if PDF is ready and not yet downloaded
                    if st.session_state["pdf_ready"] and not st.session_state["pdf_downloaded"]:
                        # Create a download button that also sets the "downloaded" flag
                        if st.download_button(
                                label="📄 Download PDF Report",
                                data=st.session_state["pdf_data"],
                                file_name="Irrigation_Report.pdf",
                                mime="application/pdf"
                        ):
                            st.session_state["pdf_downloaded"] = True  # hide button after download
                else:
                    st.error("❌ No weather data available to generate the report.")




            else:
                st.error("❌ No weather data found for this location.")
        else:
            image = Image.open("img/ExampleGraph.png")  # Assuming "images" folder in your repo
            st.image(image, caption="Example image of the graphical output", use_container_width=True)

    else:
        st.info("🖱️ Click a location on the map to begin.")