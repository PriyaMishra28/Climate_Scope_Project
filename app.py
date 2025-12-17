# ================================
# ClimateScope Dashboard
# ================================

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np

# ================================
# 1. LOAD & CLEAN DATA
# ================================

df = pd.read_csv("GlobalWeatherRepository.csv")

df.columns = df.columns.str.lower().str.strip()
df['last_updated'] = pd.to_datetime(df['last_updated'], errors='coerce')
df = df.dropna(subset=['last_updated'])

# ================================
# 2. FEATURE ENGINEERING
# ================================

df['heat_index'] = df['temperature_celsius'] + 0.33 * df['humidity'] - 0.7
df['wind_chill'] = (
    13.12
    + 0.6215 * df['temperature_celsius']
    - 11.37 * (df['wind_kph'] ** 0.16)
)

METRICS = {
    'temperature_celsius': 'Temperature (°C)',
    'humidity': 'Humidity (%)',
    'precip_mm': 'Precipitation (mm)',
    'wind_kph': 'Wind Speed (kph)',
    'heat_index': 'Heat Index',
    'wind_chill': 'Wind Chill'
}

# ================================
# 3. EXTREME EVENTS
# ================================

temp_thr = df['temperature_celsius'].quantile(0.99)
wind_thr = df['wind_kph'].quantile(0.99)

extreme_df = df[
    (df['temperature_celsius'] >= temp_thr) |
    (df['wind_kph'] >= wind_thr)
]

# ================================
# 4. DASH APP SETUP
# ================================

app = dash.Dash(__name__)
app.title = "ClimateScope Dashboard"

# ================================
# 5. HORIZONTAL FILTER BAR
# ================================

filters = html.Div(
    children=[
        html.Div([
            html.Label("Country"),
            dcc.Dropdown(
                id='country-filter',
                options=[{'label': c, 'value': c} for c in sorted(df['country'].unique())],
                multi=True,
                placeholder="All countries"
            )
        ], style={'width': '25%'}),

        html.Div([
            html.Label("Date Range"),
            dcc.DatePickerRange(
                id='date-range',
                start_date=df['last_updated'].min(),
                end_date=df['last_updated'].max()
            )
        ], style={'width': '25%'}),

        html.Div([
            html.Label("Metric"),
            dcc.Dropdown(
                id='metric-selector',
                options=[{'label': v, 'value': k} for k, v in METRICS.items()],
                value='temperature_celsius'
            )
        ], style={'width': '25%'}),

        html.Div([
            html.Label("Aggregation"),
            dcc.RadioItems(
                id='time-agg',
                options=[
                    {'label': 'Daily', 'value': 'Daily'},
                    {'label': 'Monthly', 'value': 'Monthly'}
                ],
                value='Daily',
                inline=True
            )
        ], style={'width': '25%'})
    ],
    style={
        'display': 'flex',
        'gap': '15px',
        'padding': '15px',
        'backgroundColor': '#eef5ff',
        'borderRadius': '6px',
        'marginBottom': '15px'
    }
)

# ================================
# 6. APP LAYOUT
# ================================

app.layout = html.Div([

    html.H1(
        "🌍 ClimateScope – Global Climate Analytics Dashboard",
        style={
            'textAlign': 'center',
            'padding': '15px',
            'backgroundColor': '#007ACC',
            'color': 'white'
        }
    ),

    filters,

    dcc.Tabs(id='tabs', value='exec', children=[
        dcc.Tab(label='Executive Dashboard', value='exec'),
        dcc.Tab(label='Statistical Analysis', value='stats'),
        dcc.Tab(label='Climate Trends', value='trends'),
        dcc.Tab(label='Extreme Events', value='extreme'),
        dcc.Tab(label='Help', value='help'),
    ]),

    html.Div(id='tab-content')

])

# ================================
# 7. CALLBACKS (FIXED MONTHLY AGGREGATION)
# ================================

@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value'),
    Input('country-filter', 'value'),
    Input('metric-selector', 'value'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date'),
    Input('time-agg', 'value')
)
def render_content(tab, countries, metric, start_date, end_date, time_agg):

    # If no country selected → show all
    if not countries:
        countries = df['country'].unique()

    dff = df[
        (df['country'].isin(countries)) &
        (df['last_updated'] >= start_date) &
        (df['last_updated'] <= end_date)
    ].copy()

    # ================= MONTHLY AGGREGATION FIX =================
    if time_agg == 'Monthly':
        dff['month'] = dff['last_updated'].dt.to_period('M').dt.to_timestamp()

        dff = (
            dff
            .groupby(['country', 'month'], as_index=False)
            .agg({metric: 'mean'})
            .rename(columns={'month': 'last_updated'})
        )

    # ================= EXECUTIVE DASHBOARD =================
    if tab == 'exec':
        return html.Div([
            dcc.Graph(
                figure=px.line(
                    dff,
                    x='last_updated',
                    y=metric,
                    color='country',
                    title='Trend Analysis'
                )
            ),
            dcc.Graph(
                figure=px.scatter_geo(
                    dff,
                    locations='country',
                    locationmode='country names',
                    color=metric,
                    hover_name='country',
                    title='Global Climate Map'
                )
            )
        ])

    # ================= STATISTICAL ANALYSIS =================
    if tab == 'stats':
        return html.Div([
            dcc.Graph(
                figure=px.scatter(
                    dff,
                    x='temperature_celsius',
                    y='humidity',
                    color='country',
                    title='Temperature vs Humidity'
                )
            ),
            dcc.Graph(
                figure=px.imshow(
                    dff[list(METRICS.keys())].corr(),
                    text_auto=True,
                    title='Correlation Heatmap'
                )
            )
        ])

    # ================= CLIMATE TRENDS =================
    if tab == 'trends':
        return html.Div([
            dcc.Graph(
                figure=px.area(
                    dff,
                    x='last_updated',
                    y=metric,
                    color='country',
                    title='Area Chart'
                )
            ),
            dcc.Graph(
                figure=px.violin(
                    dff,
                    y=metric,
                    box=True,
                    points='all',
                    title='Violin Plot'
                )
            ),
            dcc.Graph(
                figure=px.box(
                    dff,
                    y=metric,
                    color='country',
                    title='Box Plot'
                )
            )
        ])

    # ================= EXTREME EVENTS =================
    if tab == 'extreme':
        return html.Div([
            dash_table.DataTable(
                data=extreme_df.to_dict('records'),
                columns=[{'name': c, 'id': c} for c in extreme_df.columns],
                page_size=10,
                style_table={'overflowX': 'auto'}
            ),
            dcc.Graph(
                figure=px.histogram(
                    extreme_df,
                    x='country',
                    title='Extreme Event Frequency'
                )
            )
        ])

    # ================= HELP =================
    if tab == 'help':
        return html.Div([
            html.H4("Help & User Guide"),
            html.P(
                "Use the filters above to explore climate metrics. "
                "Choose Daily or Monthly aggregation for trend analysis. "
                "Hover, zoom, and download charts using Plotly tools."
            )
        ])

# ================================
# 8. RUN APP
# ================================

if __name__ == "__main__":
    app.run(debug=True)
