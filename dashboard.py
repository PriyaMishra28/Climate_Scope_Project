import dash
from dash import dcc, html, dash_table
import plotly.express as px
import pandas as pd

# -----------------
# 1. Data Preparation
# -----------------
try:
    df_main = pd.read_csv('GlobalWeatherRepository.csv')
except FileNotFoundError:
    print("Error: Required CSV file 'GlobalWeatherRepository.csv' not found. Please ensure it is in the same directory.")
    exit()

df_main['last_updated'] = pd.to_datetime(df_main['last_updated'])
df_main['month'] = df_main['last_updated'].dt.month

weather_cols = [
    'temperature_celsius', 'humidity', 'wind_kph', 'pressure_mb',
    'precip_mm', 'cloud', 'air_quality_PM2.5'
]

# Regional Aggregation
df_regional_agg = df_main.groupby('country').agg(
    latitude=('latitude', 'first'),
    longitude=('longitude', 'first'),
    avg_temp_celsius=('temperature_celsius', 'mean'),
    avg_wind_kph=('wind_kph', 'mean'),
    avg_humidity=('humidity', 'mean')
).reset_index()

# Monthly Aggregation
df_monthly = df_main.groupby('month')['temperature_celsius'].mean().reset_index()
df_monthly['month_name'] = df_monthly['month'].apply(lambda x: pd.to_datetime(x, format='%m').strftime('%b'))

# Correlation Matrix
df_corr_matrix = df_main[weather_cols].corr()

# Extreme Events
temp_threshold = df_main['temperature_celsius'].quantile(0.999)
wind_threshold = df_main['wind_kph'].quantile(0.999)
df_extreme = df_main[
    (df_main['temperature_celsius'] >= temp_threshold) |
    (df_main['wind_kph'] >= wind_threshold)
][['country', 'location_name', 'last_updated', 'temperature_celsius', 'wind_kph', 'condition_text', 'precip_mm']]
df_extreme['last_updated'] = df_extreme['last_updated'].dt.strftime('%Y-%m-%d %H:%M')

# -----------------
# 2. Plotly Figures
# -----------------
def create_choropleth():
    fig = px.choropleth(
        df_regional_agg,
        locations="country",
        locationmode='country names',
        color="avg_temp_celsius",
        hover_name="country",
        color_continuous_scale=px.colors.sequential.Inferno,
        title="Global Average Temperature by Country",
        template="plotly_white"
    )
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    return fig

def create_heatmap():
    fig = px.imshow(
        df_corr_matrix,
        x=weather_cols,
        y=weather_cols,
        color_continuous_scale='RdBu_r',
        aspect="auto",
        text_auto=".2f",
        title="Weather Variable Correlation Heatmap",
        template="plotly_white"
    )
    fig.update_xaxes(side="top")
    fig.update_layout(margin={"r":0,"t":80,"l":0,"b":0})
    return fig

def create_line_chart():
    fig = px.line(
        df_monthly,
        x='month_name',
        y='temperature_celsius',
        title='Seasonal Trend: Monthly Average Temperature',
        markers=True,
        template="plotly_white"
    )
    fig.update_layout(xaxis_title="Month", yaxis_title="Avg Temp")
    return fig

def create_bar_chart():
    df_top_10 = df_regional_agg.nlargest(10, 'avg_temp_celsius')
    fig = px.bar(
        df_top_10,
        x='country',
        y='avg_temp_celsius',
        title='Top 10 Hottest Countries',
        color='avg_temp_celsius',
        color_continuous_scale=px.colors.sequential.Reds,
        template="plotly_white"
    )
    fig.update_xaxes(tickangle=45)
    fig.update_layout(xaxis_title="Country", yaxis_title="Avg Temp")
    return fig

# -----------------
# 3. Dash App Layout
# -----------------
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

card_style = {
    'backgroundColor': 'white',
    'padding': '15px',
    'margin': '10px',
    'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
    'borderRadius': '5px'
}

app.layout = html.Div(style={'backgroundColor': '#f9f9f9', 'color': '#333333'}, children=[
    html.H1(
        children='Global Weather Analysis Dashboard',
        style={
            'textAlign': 'center',
            'color': '#007ACC',
            'padding': '20px 0',
            'fontSize': '28px'
        }
    ),

    # Row 1: Choropleth + Heatmap
    html.Div([
        html.Div(dcc.Graph(id='choropleth-map', figure=create_choropleth(), style={'height': '70vh'}), style=card_style, className='six columns'),
        html.Div(dcc.Graph(id='correlation-heatmap', figure=create_heatmap(), style={'height': '70vh'}), style=card_style, className='six columns'),
    ], className='row'),

    # Row 2: Line Chart + Bar Chart
    html.Div([
        html.Div(dcc.Graph(id='monthly-trend', figure=create_line_chart(), style={'height': '50vh'}), style=card_style, className='six columns'),
        html.Div(dcc.Graph(id='top-hottest', figure=create_bar_chart(), style={'height': '50vh'}), style=card_style, className='six columns'),
    ], className='row'),

    # Row 3: Extreme Events Table
    html.Div([
        html.H4("Extreme Weather Events (Top 0.1%)", style={'textAlign': 'center', 'color': '#CC0000', 'fontSize': '20px'}),
        dash_table.DataTable(
            id='extreme-events-table',
            columns=[{"name": i.replace('_', ' ').title(), "id": i} for i in df_extreme.columns],
            data=df_extreme.to_dict('records'),
            style_table={'overflowX': 'auto', 'height': '72vh', 'overflowY': 'auto'},
            style_header={'backgroundColor': '#007ACC', 'color': 'white', 'fontWeight': 'bold'},
            style_data={'backgroundColor': 'white', 'color': '#333333'},
            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'}],
            page_action="native",
            page_current=0,
            page_size=15,
        )
    ], style=card_style)
])

# -----------------
# 4. Run the App
# -----------------
if __name__ == '__main__':
    app.run(debug=True)