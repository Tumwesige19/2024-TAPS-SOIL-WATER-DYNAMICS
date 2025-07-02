# Import necessary libraries
import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="TAPS Hackathon Dashboard", layout="wide")

# Custom CSS with black sub-tab headers
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f0f0;
        font-family: 'Arial', sans-serif;
    }
    h1, h2, h3 {
        color: #4CAF50;
        text-align: center;
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 20px 0;
    }
    .metric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 15px rgba(0, 0, 0, 0.1);
    }
    footer {
        text-align: center;
        margin-top: 20px;
        font-size: 16px;
    }
    div[data-baseweb="tab-list"] button {
        color: #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Dashboard Title
st.title("2024 KSU TAPS SOIL WATER DYNAMICS")

# Sidebar Configuration
st.sidebar.header("Farm Data Overview")
st.sidebar.markdown("### 🌱 **Select Analysis Type**")
analysis_type = st.sidebar.selectbox("Choose Analysis", ["Moisture Trends", "Soil Health Overview"])

# Functions for data processing
def extract_trt_plot_summary(input_file, output_file):
    df = pd.read_excel(input_file)
    trt_plot_summary = df[['TRT_ID', 'Plot_ID', 'Block_ID']].rename(
        columns={'TRT_ID': 'Team #', 'Plot_ID': 'Plot #', 'Block_ID': 'Block #'}
    )
    trt_plot_summary.to_excel(output_file, index=False)
    return trt_plot_summary

def prepare_data(df, selected_depth):
    depth_data = df[['Date', selected_depth]].rename(columns={'Date': 'ds', selected_depth: 'y'})
    return depth_data

def forecast_moisture(depth_data, periods, freq='D'):
    model = Prophet(daily_seasonality=True)
    model.fit(depth_data)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast, model

# Load Data
input_file_path = 'C:/Users/KTM/Desktop/Hackthon_KSUTAPS/Datasets/Nutron tubes/TRT_Plot_Summary.xlsx'
output_file_path = 'C:/Users/KTM/Desktop/Hackthon_KSUTAPS/Datasets/Nutron tubes/TRT_Plot_Summary_Updated.xlsx'
trt_summary_df = extract_trt_plot_summary(input_file_path, output_file_path)

file_path = 'C:/Users/KTM/Desktop/Hackthon_KSUTAPS/Datasets/Nutron tubes/24 KSU TAPS Neutron Tube Readings_VWC.csv'
df = pd.read_csv(file_path, parse_dates=['Date'], date_parser=lambda x: pd.to_datetime(x, format='%m/%d/%Y'))

averaged_weekly_path = 'C:/Users/KTM/Desktop/Hackthon_KSUTAPS/Datasets/Sensor Data/Averaged_weekly.xlsx'
sheets = pd.read_excel(averaged_weekly_path, sheet_name=None)

# Tab Configuration
tab1, tab2, tab3, tab4 = st.tabs([
    "Tubes Readings", 
    "Acquaspy data (Moisture and EC)", 
    "Team and Plot Overview", 
    "Soil Moisture Prediction"
])

# Tab 1: Tubes Readings
with tab1:
    st.header("Moisture Changes Over Time")

    with st.sidebar.expander("Filters and Selections", expanded=True):
        selected_block = st.selectbox("Select Block #:", df['Block #'].unique())
        plots_in_block = df[df['Block #'] == selected_block]['Plot #'].unique()
        selected_plot = st.selectbox("Select Plot #:", plots_in_block)

        start_date, end_date = st.date_input("Select Date Range:", [df['Date'].min(), df['Date'].max()])

        depth_columns = [col for col in df.columns if col.startswith('V-')]
        selected_depths = st.multiselect("Select Depth(s):", depth_columns, default=depth_columns)

    filtered_df = df[
        (df['Plot #'] == selected_plot) & 
        (df['Block #'] == selected_block) &
        (df['Date'] >= pd.to_datetime(start_date)) &
        (df['Date'] <= pd.to_datetime(end_date))
    ]

    fig = go.Figure()
    for depth in selected_depths:
        fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df[depth], mode='lines+markers', name=f'{depth}'))

    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Acquaspy data (Moisture and EC)
# Tab 2: Acquaspy data (Moisture and EC)
with tab2:
    st.header("Weekly Moisture and EC Plots")

    moisture_vars = ['MS', 'M4"', 'M8"', 'M12"', 'M16"', 'M20"', 'M24"', 
                     'M28"', 'M32"', 'M36"', 'M40"', 'M44"', 'M48"']
    ec_vars = ['EC4"', 'EC8"', 'EC12"', 'EC16"', 'EC20"', 'EC24"', 
               'EC28"', 'EC32"', 'EC36"', 'EC40"', 'EC44"', 'EC48"']

    selected_teams = st.multiselect(
        "Select Team(s):", 
        options=sheets.keys(), 
        default=list(sheets.keys())[:1]
    )
    selected_measurements = st.multiselect(
        "Select Measurement Type(s):", 
        options=["Moisture", "Electrolytic Conductivity"], 
        default=["Moisture"]
    )

    moisture_selected = []
    ec_selected = []
    
    if "Moisture" in selected_measurements:
        moisture_selected = st.multiselect(
            "Select Moisture Parameters:", 
            options=moisture_vars, 
            default=["MS"]
        )
    if "Electrolytic Conductivity" in selected_measurements:
        ec_selected = st.multiselect(
            "Select EC Parameters:", 
            options=ec_vars, 
            default=["EC4\""]
        )

    fig = go.Figure()

    for team in selected_teams:
        data = sheets[team]
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])

        for parameter in moisture_selected:
            if parameter in data.columns:
                fig.add_trace(go.Scatter(
                    x=data['Timestamp'],
                    y=data[parameter],
                    mode='lines+markers',
                    name=f'{parameter} - {team}'
                ))

        for parameter in ec_selected:
            if parameter in data.columns:
                fig.add_trace(go.Scatter(
                    x=data['Timestamp'],
                    y=data[parameter],
                    mode='lines+markers',
                    name=f'{parameter} - {team}'
                ))

    fig.update_layout(
        title="Weekly Averages of Selected Moisture and EC Parameters Across Teams",
        xaxis_title="Timestamp",
        yaxis_title="Measurement Values",
        hovermode="x unified"
    )

    if fig.data:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for selected parameters. Please adjust your selections.")


# Tab 3: Team and Plot Overview
# Tab 3: Team and Plot Overview
with tab3:
    st.header("Team and Plot Overview")

    st.subheader("Select a Team to view its associated Plots and Blocks")
    selected_team = st.selectbox(
        "Select Team #:", trt_summary_df['Team #'].unique(), key="select_team_tab3"
    )

    team_data = trt_summary_df[trt_summary_df['Team #'] == selected_team]

    st.write(f"### Plots and Blocks for Team {selected_team}")
    st.dataframe(team_data[['Team #', 'Plot #', 'Block #']])

    selected_plots = st.multiselect(
        "Select Plot(s) to Visualize Moisture Content Trends:",
        options=team_data['Plot #'].unique(),
        default=team_data['Plot #'].unique(),
        key="plots_tab3"
    )

    depth_columns = [col for col in df.columns if col.startswith('V-')]
    selected_depths = st.multiselect(
        "Select Depth(s):", depth_columns, default=depth_columns, key="depths_tab3"
    )

    start_date, end_date = st.date_input(
        "Select Date Range:",
        [df['Date'].min(), df['Date'].max()],
        key="date_range_tab3"
    )

    filtered_df_team = df[
        (df['Plot #'].isin(selected_plots)) &
        (df['Date'] >= pd.to_datetime(start_date)) &
        (df['Date'] <= pd.to_datetime(end_date))
    ]

    st.write("### Filtered Data for Selected Plots and Date Range")
    st.dataframe(filtered_df_team)

    if filtered_df_team.empty:
        st.warning("No data available for the selected filters.")
    else:
        st.subheader(f"Moisture Content Trend for Team {selected_team}")

        fig = go.Figure()
        for plot in selected_plots:
            plot_data = filtered_df_team[filtered_df_team['Plot #'] == plot]
            for depth in selected_depths:
                if depth in plot_data.columns:
                    fig.add_trace(go.Scatter(
                        x=plot_data['Date'],
                        y=plot_data[depth],
                        mode='lines+markers',
                        name=f'Plot {plot} - Depth {depth}'
                    ))

        fig.update_layout(
            title=f'Soil Moisture Trend for Team {selected_team} Across Selected Plots and Depths',
            xaxis_title='Date',
            yaxis_title='Volumetric Water Content',
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)


# Tab 4: Soil Moisture Prediction
with tab4:
    st.header("Soil Moisture Prediction")
    selected_depth = st.selectbox("Select Depth:", depth_columns)
    forecast_horizon = st.slider("Forecast Horizon (days):", 1, 30, 7)

    depth_data = prepare_data(df, selected_depth)
    forecast, _ = forecast_moisture(depth_data, periods=forecast_horizon)

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
    fig4.add_trace(go.Scatter(x=depth_data['ds'], y=depth_data['y'], mode='lines', name='Actual'))
    st.plotly_chart(fig4, use_container_width=True)

# Footer
st.markdown("""
<footer>
    Developed by The Chefs Team | 
    <a href="https://github.com/your-repo">GitHub</a> | 
    <a href="https://example.com/contact">Contact Us</a>
</footer>
""", unsafe_allow_html=True)


#       streamlit run "f:\TAPS Dashboard\2024-TAPS-SOIL-WATER-DYNAMICS\Dashboard_Taps.py"