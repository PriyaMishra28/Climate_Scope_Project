import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# =================================================================
# 1. DATA PREPROCESSING & FEATURE ENGINEERING
# =================================================================

def load_data():
    # Load your specific file
    df = pd.read_csv("GlobalWeatherRepository.csv")
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()
    df['last_updated'] = pd.to_datetime(df['last_updated'])
    
    # 1.1 Feature Engineering: Heat Index & Wind Chill
    df['heat_index'] = df['temperature_celsius'] + 0.33 * df['humidity'] - 0.7
    df['wind_chill'] = 13.12 + 0.6215 * df['temperature_celsius'] - 11.37 * (df['wind_kph']**0.16)
    
    # 1.2 Moving Averages (Calculated per country)
    df = df.sort_values(['country', 'last_updated'])
    df['temp_7day_avg'] = df.groupby('country')['temperature_celsius'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    
    # 1.3 Seasonal Mapping
    df['month'] = df['last_updated'].dt.month
    df['season'] = df['month'].map({
        12:'Winter', 1:'Winter', 2:'Winter',
        3:'Spring', 4:'Spring', 5:'Spring',
        6:'Summer', 7:'Summer', 8:'Summer',
        9:'Autumn', 10:'Autumn', 11:'Autumn'
    })
    
    # 1.4 ISO-3 Country Code Mapping (Sample dictionary - expand as needed)
    # Note: In a real project, use 'pycountry' library for 100% coverage
    iso_mapping = {
        "Afghanistan": "AFG", "Brazil": "BRA", "Canada": "CAN", "China": "CHN",
        "France": "FRA", "Germany": "DEU", "India": "IND", "Japan": "JPN",
        "Mexico": "MEX", "Russia": "RUS", "United Kingdom": "GBR", "USA": "USA", "United States of America": "USA"
    }
    df['iso_alpha'] = df['country'].map(iso_mapping).fillna("")
    
    return df

df = load_data()

# =================================================================
# 2. UI COMPONENTS & LAYOUT
# =================================================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True)
app.title = "ClimateScope Analytics"

# Sidebar Styling
SIDEBAR_STYLE = {
    "position": "fixed", "top": 0, "left": 0, "bottom": 0,
    "width": "22rem", "padding": "2rem 1rem", "background-color": "#f8f9fa",
    "overflow-y": "auto", "border-right": "1px solid #dee2e6"
}

CONTENT_STYLE = {"margin-left": "24rem", "margin-right": "2rem", "padding": "2rem 1rem"}



sidebar = html.Div([
    html.H2("ClimateScope", className="text-primary fw-bold"),
    html.Hr(),
    html.P("Global Climate Intelligence", className="text-muted small"),
    
    html.Div([
        html.Label("🌍 Country Selection", className="fw-bold"),
        dcc.Dropdown(
            id='country-filter',
            options=[{'label': c, 'value': c} for c in sorted(df['country'].unique())],
            multi=True, placeholder="Select Countries (Global by default)"
        ),
    ], className="mb-4"),

    html.Div([
        html.Label("📅 Date Range", className="fw-bold"),
        dcc.DatePickerRange(
            id='date-filter',
            min_date_allowed=df['last_updated'].min(),
            max_date_allowed=df['last_updated'].max(),
            start_date=df['last_updated'].min(),
            end_date=df['last_updated'].max(),
            style={'width': '100%'}
        ),
    ], className="mb-4"),

    html.Div([
        html.Label("📊 Primary Metric", className="fw-bold"),
        dcc.Dropdown(
            id='metric-selector',
            options=[
                {'label': 'Temperature (°C)', 'value': 'temperature_celsius'},
                {'label': 'Humidity (%)', 'value': 'humidity'},
                {'label': 'Precipitation (mm)', 'value': 'precip_mm'},
                {'label': 'Wind Speed (kph)', 'value': 'wind_kph'},
                {'label': 'Heat Index', 'value': 'heat_index'},
                {'label': '7-Day Moving Avg (Temp)', 'value': 'temp_7day_avg'}
            ], value='temperature_celsius'
        ),
    ], className="mb-4"),

    html.Div([
        html.Label("⚠️ Extreme Event Threshold", className="fw-bold"),
        dcc.Slider(id='threshold-slider', min=0, max=50, step=1, value=35, 
                   marks={0:'0', 25:'25°C', 35:'35°C', 50:'50'}),
        html.Small("Set threshold for Heatwave/Extreme detection", className="text-muted")
    ], className="mb-4"),

    dbc.Checklist(
        options=[{"label": "Enable Z-Score Normalization", "value": 1}],
        id="norm-toggle", switch=True, className="mb-4"
    ),

    html.Hr(),
    html.Div([
        html.P("Built by Priya Mishra", className="small text-muted mb-0"),
        html.A("GitHub Profile", href="https://github.com/priyamishra", target="_blank", className="small")
    ])
], style=SIDEBAR_STYLE)

content = html.Div([
    dbc.Tabs([
        dbc.Tab(label="Executive Summary", tab_id="exec"),
        dbc.Tab(label="Statistical Deep-Dive", tab_id="stats"),
        dbc.Tab(label="Climate Trends", tab_id="trends"),
        dbc.Tab(label="Extreme Events", tab_id="extreme"),
        dbc.Tab(label="User Guide", tab_id="help"),
    ], id="tabs", active_tab="exec", className="mb-4"),
    
    html.Div(id="tab-content")
], style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

# =================================================================
# 3. ANALYTICS & CALLBACK LOGIC
# =================================================================

@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab"),
     Input("country-filter", "value"),
     Input("metric-selector", "value"),
     Input("date-filter", "start_date"),
     Input("date-filter", "end_date"),
     Input("norm-toggle", "value"),
     Input("threshold-slider", "value")]
)
def render_tab_content(active_tab, countries, metric, start, end, norm, threshold):
    # Filter DataFrame
    dff = df[(df['last_updated'] >= start) & (df['last_updated'] <= end)].copy()
    if countries:
        dff = dff[dff['country'].isin(countries)]
    
    # Handle Normalization
    display_col = 'val_to_plot'
    if norm:
        mean, std = dff[metric].mean(), dff[metric].std()
        dff[display_col] = (dff[metric] - mean) / (std if std != 0 else 1)
        y_label = f"{metric} (Z-Score)"
    else:
        dff[display_col] = dff[metric]
        y_label = metric.replace('_', ' ').title()

    # --- TAB 1: EXECUTIVE DASHBOARD ---
    if active_tab == "exec":
        
        avg_val = dff[metric].mean()
        max_val = dff[metric].max()
        
        return html.Div([
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([html.H6("Avg Analysis Value"), html.H3(f"{avg_val:.2f}")]), color="primary", outline=True), width=4),
                dbc.Col(dbc.Card(dbc.CardBody([html.H6("Peak Value Recorded"), html.H3(f"{max_val:.2f}")]), color="danger", outline=True), width=4),
                dbc.Col(dbc.Card(dbc.CardBody([html.H6("Active Monitoring"), html.H3(f"{len(dff['country'].unique())} Regions")]), color="success", outline=True), width=4),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col(dcc.Graph(figure=px.choropleth(
                    dff, locations="iso_alpha", color=display_col, hover_name="country",
                    title="Global Spatial Variation", color_continuous_scale="Viridis"
                )), width=12),
            ], className="mb-4"),
            
            dcc.Graph(figure=px.line(dff, x="last_updated", y=display_col, color="country", title="Temporal Snapshot").update_xaxes(rangeslider_visible=True))
        ])

    # --- TAB 2: STATISTICAL ANALYSIS ---
    elif active_tab == "stats":
        
        corr_data = dff[['temperature_celsius', 'humidity', 'wind_kph', 'precip_mm', 'heat_index']].corr()
        return dbc.Row([
            dbc.Col(dcc.Graph(figure=px.imshow(corr_data, text_auto=True, title="Inter-variable Correlation")), width=6),
            dbc.Col(dcc.Graph(figure=px.box(dff, x="country", y=display_col, color="country", title="Regional Variance (Box Plot)")), width=6),
            dbc.Col(dcc.Graph(figure=px.scatter(dff, x="temperature_celsius", y="humidity", color="country", size="wind_kph", title="Atmospheric Relationship (Bubble Chart)")), width=12)
        ])

    # --- TAB 3: CLIMATE TRENDS ---
    elif active_tab == "trends":
        
        return html.Div([
            dcc.Graph(figure=px.area(dff, x="last_updated", y=display_col, color="country", title="Cumulative Trend Analysis")),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=px.histogram(dff, x=display_col, color="season", marginal="rug", title="Seasonal Distribution Density")), width=12)
            ])
        ])

    # --- TAB 4: EXTREME EVENTS ---
    elif active_tab == "extreme":
        extreme_events = dff[dff['temperature_celsius'] >= threshold]
        top_5 = dff.nlargest(5, 'temperature_celsius')[['last_updated', 'country', 'location_name', 'temperature_celsius']]
        
        return html.Div([
            html.H4(f"Extreme Events Identified (Threshold: {threshold})", className="text-danger"),
            dash_table.DataTable(
                data=top_5.to_dict('records'),
                columns=[{"name": i.replace('_',' ').title(), "id": i} for i in top_5.columns],
                style_header={'backgroundColor': '#f8d7da', 'fontWeight': 'bold'},
                style_cell={'textAlign': 'left', 'padding': '10px'}
            ),
            dcc.Graph(figure=px.bar(
                extreme_events.groupby('country').size().reset_index(name='count'), 
                x='country', y='count', title="Frequency of Threshold Violations by Country", color_discrete_sequence=['#dc3545']
            ))
        ])

    # --- TAB 5: HELP ---
    elif active_tab == "help":
        return dbc.Card([
            dbc.CardBody([
                html.H4("How to use ClimateScope"),
                html.Ul([
                    html.Li("Use the Sidebar to filter by specific countries or timeframes."),
                    html.Li("Switch metrics to see how Humidity or Wind impact the Heat Index."),
                    html.Li("Enable Normalization to compare countries with vastly different climates on a scale of 0 to 1."),
                    html.Li("The Extreme Events tab highlights areas exceeding your custom safety threshold."),
                    html.Li("All charts are interactive: hover for data, drag to zoom, and click the camera icon to download as PNG.")
                ])
            ])
        ], className="mt-4")

# =================================================================
# 4. EXECUTION
# =================================================================

if __name__ == "__main__":
    app.run(debug=True, port=8051)