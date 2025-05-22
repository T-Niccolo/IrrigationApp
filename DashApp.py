import dash
from dash import html, dcc, Output, Input, State, dash_table
import dash_leaflet as dl
import ee
import pandas as pd
import datetime
import plotly.graph_objects as go
import base64
import numpy as np
import os


# Initialize Earth Engine
# service_account = 'dash-bloom@ee-mzwienie.iam.gserviceaccount.com'
# credentials = ee.ServiceAccountCredentials(service_account, '.new_private_key.json')
ee.Initialize()

print("Earth Engine initialized")

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

map_center = [35.26, -119.15]

app.layout = html.Div([
   

    dcc.Markdown("""
For further information contact: - [**Or Sperling**](mailto:orsp@volcani.agri.gov.il) (ARO-Volcani)  or [**Maciej Zwieniecki**](mailto:mzwienie@ucdavis.edu) (UC Davis)  
Contributors: - [Zack Ellis](mailto:zellis@ucdavis.edu) (UC Davis) and [Niccolò Tricerri](mailto:niccolo.tricerri@unito.it) (UNITO - IUSS Pavia)
""", style={"textAlign": "center", "fontFamily": "Arial"}),

    # Responsive layout for map + sliders and output
    html.Div([
        html.Div([
            html.H4("Map View - use to select your orchard (move, zoom and click)"),
            dl.Map(center=map_center, zoom=12, children=[
                dl.TileLayer(url="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
                             attribution="Google Satellite"),
                dl.LayerGroup(id="click-marker")
            ], style={"height": "400px"}, id="map"),
    
            html.H4("Irrigation Parameters - please adjust using sliders"),
            html.Label("Units"),
            dcc.RadioItems(
                id="unit-toggle",
                options=[
                    {"label": "Millimeters (mm)", "value": "mm"},
                    {"label": "Inches (in)", "value": "in"}
                ],
                value="mm",
                inline=True
            ),
            html.Br(),
    
            html.Div([
                html.Label("Winter Irrigation   ", style={"marginBottom": "5px"}),
                # dcc.Input(id="input-winter", type="number", value=0, style={"width": "80px", "marginBottom": "5px"}),
                dcc.Slider(id="slider-winter", min=0, max=700, step=50, value=0)
            ], style={"marginBottom": "15px"}),
    
            html.Br(),
            html.Label("Irrigation Months"),
            dcc.RangeSlider(id="slider-months", min=1, max=12, step=1, value=[5, 10],
                            marks={i: str(i) for i in range(1, 13)}),
    
            html.Br(),
        html.Div([
            html.Label("Water Allocation - enter data or use slider   ", style={"marginBottom": "5px"}),
            # dcc.Input(id="input-allocation", type="number", value=600, style={"width": "80px", "marginBottom": "5px"}),
            dcc.Slider(id="slider-allocation", min=0, max=1000, step=10, value=600,
                       marks={i: str(i) for i in range(0, 1001, 200)})
        ], style={"marginBottom": "15px"}),
    
        ], className="responsive-column"),
    
        # Stores
        dcc.Store(id="unit-store", data="mm"),
        dcc.Store(id="click-coords"),
        dcc.Store(id="rainfall-store", data=0),
    
        # Output section
        html.Div([
            html.Div(id="output", style={"marginTop": "10px", "textAlign": "center"}),
            dcc.Graph(id="chart"),
            html.Div(id="data-table-container"),
            html.A("Download CSV", id="download-link", download="irrigation_data.csv",
                   href="", target="_blank", style={"display": "none"})
        ], className="responsive-column")
    ], className="responsive-container"),
    
  
])

                                  
# === Earth Engine data functions ===
def fetch_ndvi(lat, lon):
    poi = ee.Geometry.Point([lon, lat])
    start = f"{datetime.datetime.now().year - 1}-05-01"
    end = f"{datetime.datetime.now().year - 1}-06-01"
    img = ee.ImageCollection('COPERNICUS/S2_HARMONIZED').filterDate(start, end).median()
    ndvi = img.normalizedDifference(['B8', 'B4']).reduceRegion(ee.Reducer.mean(), poi, scale=50).get('nd')
    return ndvi.getInfo()

def fetch_rainfall(lat, lon):
    today = datetime.datetime.now()
    start = f"{today.year - 1}-11-01"
    end = today.strftime("%Y-%m-%d")
    point = ee.Geometry.Point(lon, lat)
    rain = ee.ImageCollection("OREGONSTATE/PRISM/AN81d").filterDate(start, end).select("ppt").sum()
    return rain.reduceRegion(ee.Reducer.first(), point, 1000).get("ppt").getInfo()

def fetch_et0(lat, lon):
    today = datetime.datetime.now()
    start = ee.Date(f"{today.year - 1}-01-01")
    point = ee.Geometry.Point(lon, lat)
    months = ee.List.sequence(0, 11)

    def monthly(n):
        s = start.advance(n, "month")
        e = s.advance(1, "month")
        et = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterDate(s, e).select("eto").sum()
        val = et.reduceRegion(ee.Reducer.first(), point, 4000).get("eto")
        return ee.Feature(None, {"month": s.format("M"), "et0": val})

    fc = ee.FeatureCollection(months.map(monthly)).getInfo()["features"]
    return pd.DataFrame([{"month": int(f["properties"]["month"]), "ET₀": f["properties"]["et0"]} for f in fc if f["properties"]["et0"] is not None])

# === Irrigation calculation ===
def calculate_irrigation(df, ndvi, rain, winter, months, allocation):

    print("Calculating irrigation...")
    
    df = df[df["month"].between(2, 11)]
    df = pd.merge(pd.DataFrame({"month": range(2, 12)}), df, on="month", how="left")
    df["ET₀"] = pd.to_numeric(df["ET₀"], errors="coerce").fillna(0) * 0.8
    df["ET1"] = df["ET₀"] * (ndvi / 0.7 if ndvi < 0.67 else ndvi)

    grow_months = list(range(months[0], months[1] + 1))
    SWI = (rain + winter - df.loc[~df['month'].isin(grow_months), 'ET1'].sum() - 50) / len(grow_months)
    df["irrigation"] = 0
    df.loc[df["month"].isin(grow_months), "irrigation"] = df["ET1"] - SWI
    df["irrigation"] = df["irrigation"].clip(lower=0)
    
    vst = df.loc[df['month'] == 7, 'irrigation'] * 0.1
    df.loc[df['month'] == 7, 'irrigation'] *= 0.8
    df.loc[df['month'].isin([8, 9]), 'irrigation'] += vst.values[0] if not vst.empty else 0
      
    factor = allocation / df.loc[df["month"].isin(grow_months), "irrigation"].sum() if df["irrigation"].sum() > 0 else 1
    # df["irrigation"] *= factor
    df["SW1"] = (rain + winter - df["ET1"].cumsum() + df["irrigation"].cumsum()).clip(lower=0)
    df["alert"] = np.where(df["SW1"] == 0, "drought", "safe")
    df["month"] = pd.to_datetime(df["month"], format='%m').dt.strftime('%B')
    return df.round(2)


# === Slider range updates ===
# === Slider range updates with rainfall init merged ===
@app.callback(
    Output("slider-winter", "min"), Output("slider-winter", "max"),
    Output("slider-winter", "step"), Output("slider-winter", "marks"),
    Output("slider-winter", "value"), 

    Output("slider-allocation", "min"), Output("slider-allocation", "max"),
    Output("slider-allocation", "step"), Output("slider-allocation", "marks"),
    Output("slider-allocation", "value"), 

    Input("unit-toggle", "value"),
    Input("rainfall-store", "data"),
    Input("slider-winter", "value"),
    Input("slider-allocation", "value"),
    prevent_initial_call=True
)
def update_sliders_and_inputs(unit, rainfall, slider_winter, slider_allocation):
    ctx = dash.callback_context
    triggered = ctx.triggered_id

 
    if triggered == "unit-toggle":

        unit_conversion=0.0393701 if unit == "in" else 1

        winter_val = int(round(slider_winter*unit_conversion))
        allocation_val = int(round(slider_allocation*unit_conversion))
        
     # Define slider ranges
    if unit == "mm":
        winter_cfg = (0, 700, 10, {i: str(i) for i in range(0, 701, 300)})
        alloc_cfg = (0, 1500, 10, {i: str(i) for i in range(0, 1601, 300)})
    else:
        winter_cfg = (0, 30, 1, {i: str(i) for i in range(0, 31, 10)})
        alloc_cfg = (0, 55, 1, {i: str(i) for i in range(0, 56, 10)})

    return (
        *winter_cfg, winter_val, 
        *alloc_cfg, allocation_val,
    )


# === Add a callback to store click coords and rainfall ===
@app.callback(
    Output("click-coords", "data"),
    Output("rainfall-store", "data"),
    Input("map", "click_lat_lng"),
    prevent_initial_call=True
)
def store_coords_and_rainfall(click_lat_lng):
    if not click_lat_lng:
        return dash.no_update, dash.no_update
    lat, lon = click_lat_lng
    try:
        rain = fetch_rainfall(lat, lon)
        return {"lat": lat, "lon": lon}, rain
    except Exception as e:
        print("Error fetching rainfall:", str(e))
        return dash.no_update, dash.no_update

# === Update slider value on location select ===


# === Output, Graph, Table ===

@app.callback(
    Output("output", "children"),
    Output("click-marker", "children"),
    Output("chart", "figure"),
    Output("download-link", "href"),
    Output("download-link", "style"),
    Output("data-table-container", "children"),
    Input("map", "click_lat_lng"),
    Input("slider-winter", "value"),
    Input("slider-months", "value"),
    Input("slider-allocation", "value"),
    Input("unit-toggle", "value"),
    State("click-marker", "children")
)
def update_output(click_lat_lng, winter, months, allocation, unit, marker_state):
    if not click_lat_lng:
        if marker_state:
            click_lat_lng = marker_state[0]["props"]["position"]
        else:
            return "Click on the map to get data.", [], {}, "", {"display": "none"}, ""

    lat, lon = click_lat_lng
    print(f"Clicked location: ({lat}, {lon})")
    try:
        ndvi = fetch_ndvi(lat, lon)
        rain = fetch_rainfall(lat, lon)
        et0_df = fetch_et0(lat, lon)

        unit_label = "inches" if unit == "in" else "mm"
        conversion = 0.0393701 if unit == "in" else 1

        df = calculate_irrigation(et0_df, ndvi, rain, winter/conversion, months, allocation/conversion)

        df["ET₀"] = (df["ET₀"] * conversion).round(1)
        df["ET1"] = (df["ET1"] * conversion).round(1)
        df["irrigation"] = (df["irrigation"] * conversion).round(1)
        df["SW1"] = (df["SW1"] * conversion).round(1)
        y_title = f"Water ({unit_label})"

       
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["month"], y=df["irrigation"], name="Irrigation",
            marker_color=["#FF4B4B" if a == "drought" else "#3897c5" for a in df["alert"]]
        ))
        fig.add_trace(go.Scatter(
            x=df["month"], y=df["SW1"], name="Soil Water",
            mode="lines+markers", line=dict(color="#74ac72")
        ))
        fig.add_trace(go.Scatter(
            x=df["month"], y=df["ET₀"], name="ET₀",
            mode="lines+markers", line=dict(color="#ffa500")
        ))

        fig.update_layout(
            title="Monthly Irrigation Plan",
            xaxis_title="Month",
            yaxis_title=y_title,
            template="plotly_white"
)
        df_display = df.drop(columns=df.columns[2])

        table = dash_table.DataTable(
            columns=[{"name": col, "id": col} for col in df_display.columns],
            data=df_display.to_dict("records"),
            style_cell={"textAlign": "center"},
            style_header={"backgroundColor": "#f0f0f0", "fontWeight": "bold"}
        )

        csv_str = df.to_csv(index=False)
        b64 = base64.b64encode(csv_str.encode()).decode()
        href = f"data:text/csv;base64,{b64}"

        rain_display = rain*conversion if unit == "in" else rain
        rain_unit = "in" if unit == "in" else "mm"
        msg = f"📍 Location: ({lat:.3f}, {lon:.3f}) | NDVI: {ndvi:.2f} | Rainfall: {rain_display:.1f} {rain_unit}"
        marker = dl.Marker(position=[lat, lon], children=dl.Tooltip("Selected Location"))

        return msg, [marker], fig, href, {"display": "inline-block"}, table

    except Exception as e:
        return f"⚠️ Error: {str(e)}", [], {}, "", {"display": "none"}, ""

if __name__ == "__main__":
    app.run(debug=True)


#    port = int(os.environ.get("PORT", 8050))
#    app.run(debug=False, host="0.0.0.0", port=port)