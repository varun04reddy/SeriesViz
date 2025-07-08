import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import io
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="SeriesViz - Time Series Data Visualization",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def load_data(uploaded_file):
    """Load CSV file and return DataFrame"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            st.error("Please upload a CSV file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def convert_to_datetime(df, column):
    """Convert column to datetime format"""
    try:
        # First, try to convert directly
        df[column] = pd.to_datetime(df[column])
        return df
    except Exception as e:
        # If that fails, try removing timezone information
        try:
            # Convert to string and remove timezone info if present
            df[column] = df[column].astype(str).str.replace(r'\+00:00$', '', regex=True)
            df[column] = df[column].str.replace(r'\+00$', '', regex=True)
            df[column] = df[column].str.replace(r'Z$', '', regex=True)
            
            # Now convert to datetime
            df[column] = pd.to_datetime(df[column])
            return df
        except Exception as e2:
            st.error(f"❌ Both conversion methods failed for {column}: {str(e2)}")
            return df

def remove_nan_values(df, columns):
    """Remove NaN values from specified columns"""
    df_clean = df.copy()
    for col in columns:
        df_clean = df_clean.dropna(subset=[col])
    return df_clean

def z_score_normalize(df, columns):
    """Z-score normalize specified columns"""
    df_normalized = df.copy()
    for col in columns:
        if col in df_normalized.columns:
            df_normalized[col] = stats.zscore(df_normalized[col].dropna())
    return df_normalized

def calculate_rolling_correlation(df, columns, window_size):
    """Calculate rolling correlation between variables"""
    if len(columns) < 2:
        return None
    
    correlations = {}
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns[i+1:], i+1):
            if col1 in df.columns and col2 in df.columns:
                # Calculate rolling correlation
                rolling_corr = df[col1].rolling(window=window_size).corr(df[col2])
                correlations[f"{col1} vs {col2}"] = rolling_corr
    
    return correlations

def create_time_series_plot(df, timestamp_col, variable_cols, plot_type="line", title="Time Series Plot"):
    """Create time series plot using Plotly"""
    fig = go.Figure()
    
    for col in variable_cols:
        if col in df.columns:
            if plot_type == "scatter":
                fig.add_trace(
                    go.Scatter(
                        x=df[timestamp_col],
                        y=df[col],
                        mode='markers',
                        name=col,
                        marker=dict(size=4)
                    )
                )
            else:  # line plot
                fig.add_trace(
                    go.Scatter(
                        x=df[timestamp_col],
                        y=df[col],
                        mode='lines',
                        name=col,
                        line=dict(width=2)
                    )
                )
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode='x unified',
        height=500,
        showlegend=True,
        template="plotly_white"
    )
    
    return fig

def create_correlation_plot(correlations, df, timestamp_col, plot_type="line", title="Rolling Correlation"):
    """Create correlation plot using Plotly"""
    if not correlations:
        return None
    
    fig = go.Figure()
    
    for corr_name, corr_values in correlations.items():
        if corr_values is not None and not corr_values.isna().all():
            # Get the corresponding timestamp values for the correlation data
            # Use the index of the correlation series to get the right timestamps
            # Create a mapping from the correlation series index to timestamp values
            timestamp_values = df[timestamp_col].reindex(corr_values.index)
            
            if plot_type == "scatter":
                fig.add_trace(
                    go.Scatter(
                        x=timestamp_values,
                        y=corr_values.values,
                        mode='markers',
                        name=corr_name,
                        marker=dict(size=4)
                    )
                )
            else:  # line plot
                fig.add_trace(
                    go.Scatter(
                        x=timestamp_values,
                        y=corr_values.values,
                        mode='lines',
                        name=corr_name,
                        line=dict(width=2)
                    )
                )
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Correlation Coefficient",
        hovermode='x unified',
        height=500,
        showlegend=True,
        template="plotly_white"
    )
    
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    return fig

def create_large_plot_display(fig, title, plot_type):
    """Create a larger plot display in a separate tab"""
    st.markdown(f"### 📊 {title} - Large View")
    st.markdown(f"*Plot Type: {plot_type.title()}*")
    
    # Make the plot larger
    fig.update_layout(
        height=700,  # Increased height
        width=None,  # Full width
        margin=dict(l=50, r=50, t=80, b=50),  # Better margins
        font=dict(size=14)  # Larger font
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
    
    # Add download options
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download as PNG",
            data=fig.to_image(format="png", width=1200, height=800),
            file_name=f"{title.lower().replace(' ', '_')}_{plot_type}.png",
            mime="image/png"
        )
    with col2:
        st.download_button(
            label="📥 Download as HTML",
            data=fig.to_html(include_plotlyjs=True),
            file_name=f"{title.lower().replace(' ', '_')}_{plot_type}.html",
            mime="text/html"
        )

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 SeriesViz</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Time Series Data Visualization Dashboard</p>', unsafe_allow_html=True)
    
    # Sidebar for file upload
    with st.sidebar:
        st.markdown('<h3 class="section-header">📁 Data Upload</h3>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your time series data in CSV format"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Display file info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.2f} KB",
                "File type": uploaded_file.type
            }
            st.json(file_details)
    
    # Main content area
    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)
        
        if df is not None:
            # Data Overview Section
            st.markdown('<h2 class="section-header">📋 Data Overview</h2>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")
            with col4:
                missing_values = df.isnull().sum().sum()
                st.metric("Missing Values", missing_values)
            
            # Display first few rows
            st.subheader("Preview of Data")
            st.dataframe(df.head(), use_container_width=True)
            
            # Column Information
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True)
            
            # Data Configuration Section
            st.markdown('<h2 class="section-header">⚙️ Data Configuration</h2>', unsafe_allow_html=True)
            
            # Column selection
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Timestamp Column")
                timestamp_col = st.selectbox(
                    "Select the timestamp column:",
                    options=df.columns.tolist(),
                    help="Choose the column that contains time/date information"
                )
                
                if timestamp_col:
                    # Convert to datetime
                    df = convert_to_datetime(df, timestamp_col)
                    
                    # Check if conversion was successful
                    try:
                        # Test if the column is now datetime
                        pd.to_datetime(df[timestamp_col].iloc[0])
                        st.success(f"✅ Successfully converted {timestamp_col} to datetime format")
                        st.info(f"📅 Time range: {df[timestamp_col].min()} to {df[timestamp_col].max()}")
                    except:
                        st.warning("⚠️ Unable to convert timestamp column to datetime format.")
            
            with col2:
                st.subheader("Variable Columns")
                variable_cols = st.multiselect(
                    "Select variable columns to analyze:",
                    options=[col for col in df.columns if col != timestamp_col],
                    help="Choose the columns you want to visualize and analyze"
                )
            
            # Time Range Filter Section (separate from column selection)
            if timestamp_col:
                # Check if the timestamp column is properly converted to datetime
                try:
                    # Test if we can access datetime properties
                    min_time = df[timestamp_col].min().to_pydatetime()
                    max_time = df[timestamp_col].max().to_pydatetime()
                    
                    st.markdown('<h3 class="section-header">⏰ Time Range Filter</h3>', unsafe_allow_html=True)
                    
                    # Create slider for time range
                    time_range = st.slider(
                        "Select time range to analyze:",
                        min_value=min_time,
                        max_value=max_time,
                        value=(min_time, max_time),
                        format="YYYY-MM-DD HH:MM",
                        help="Drag the sliders to select a specific time range for analysis"
                    )
                    
                    # Filter data based on selected time range
                    df_filtered = df[(df[timestamp_col] >= pd.Timestamp(time_range[0])) & (df[timestamp_col] <= pd.Timestamp(time_range[1]))]
                    st.info(f"📊 Filtered data: {len(df_filtered)} rows from {time_range[0]} to {time_range[1]}")
                except:
                    # If we can't access datetime properties, use unfiltered data
                    df_filtered = df.copy()
                    st.warning("⚠️ Time range filter not available - using full dataset")
            else:
                df_filtered = df.copy()
            
            # Data Preprocessing Section
            if variable_cols:
                st.markdown('<h2 class="section-header">🔧 Data Preprocessing</h2>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Data Cleaning")
                    remove_nan = st.checkbox(
                        "Remove NaN values from variable columns",
                        value=True,
                        help="Remove rows with missing values in selected variable columns"
                    )
                    
                    if remove_nan:
                        df_processed = remove_nan_values(df_filtered, variable_cols)
                        st.success(f"✅ Removed NaN values. Remaining rows: {len(df_processed)}")
                    else:
                        df_processed = df_filtered.copy()
                
                with col2:
                    st.subheader("Data Normalization")
                    normalize_data = st.checkbox(
                        "Z-score normalize variable columns",
                        help="Standardize variables to have mean=0 and std=1"
                    )
                    
                    if normalize_data:
                        df_processed = z_score_normalize(df_processed, variable_cols)
                        st.success("✅ Variables normalized using Z-score")
                
                # Visualization Section
                st.markdown('<h2 class="section-header">📈 Visualizations</h2>', unsafe_allow_html=True)
                
                # Plot type selection
                col1, col2 = st.columns(2)
                with col1:
                    time_series_plot_type = st.selectbox(
                        "Time Series Plot Type:",
                        options=["line", "scatter"],
                        help="Choose between line plot or scatter plot for time series"
                    )
                with col2:
                    correlation_plot_type = st.selectbox(
                        "Correlation Plot Type:",
                        options=["line", "scatter"],
                        help="Choose between line plot or scatter plot for correlations"
                    )
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["📊 Regular View", "📅 Day-Specific Analysis", "🕐 Hourly Bucketing Analysis"])
                
                with tab1:
                    # Regular view with smaller plots
                    st.subheader("Time Series Plot")
                    if timestamp_col and variable_cols:
                        time_series_fig = create_time_series_plot(
                            df_processed, 
                            timestamp_col, 
                            variable_cols,
                            time_series_plot_type,
                            "Time Series of Selected Variables"
                        )
                        st.plotly_chart(time_series_fig, use_container_width=True)
                    
                    # Correlation Analysis
                    st.subheader("Rolling Correlation Analysis")
                    
                    # Controls in a single row
                    col1, col2, col3 = st.columns([1, 1, 2])
                    
                    with col1:
                        show_correlation = st.checkbox(
                            "Show rolling correlation",
                            help="Display correlation between variables over time"
                        )
                    
                    with col2:
                        if show_correlation and len(variable_cols) >= 2:
                            window_size = st.number_input(
                                "Correlation window size:",
                                min_value=2,
                                max_value=min(100, len(df_processed) // 10),
                                value=min(20, len(df_processed) // 20),
                                help="Number of data points to include in correlation calculation",
                                key="window_size_regular"
                            )
                    
                    # Plot in full width below controls
                    if show_correlation and len(variable_cols) >= 2:
                        # Calculate correlations
                        correlations = calculate_rolling_correlation(df_processed, variable_cols, window_size)
                        
                        if correlations:
                            correlation_fig = create_correlation_plot(
                                correlations, df_processed, timestamp_col, correlation_plot_type,
                                f"Rolling Correlation (Window Size: {window_size})"
                            )
                            st.plotly_chart(correlation_fig, use_container_width=True)
                        else:
                            st.warning("⚠️ Unable to calculate correlations. Check your data.")
                    elif show_correlation:
                        st.warning("⚠️ Need at least 2 variable columns for correlation analysis.")
                
                with tab2:
                    # Day-Specific Analysis
                    st.subheader("📅 Day-Specific Analysis")
                    st.markdown("Select multiple dates to compare intraday movements and patterns.")
                    
                    if timestamp_col and variable_cols:
                        # Get unique dates from the data
                        df_processed[timestamp_col] = pd.to_datetime(df_processed[timestamp_col])
                        unique_dates = df_processed[timestamp_col].dt.date.unique()
                        unique_dates = sorted(unique_dates)
                        
                        if len(unique_dates) > 0:
                            # Multiple date selection
                            col1, col2 = st.columns(2)
                            with col1:
                                selected_dates = st.multiselect(
                                    "Select dates to analyze:",
                                    options=unique_dates,
                                    format_func=lambda x: x.strftime('%Y-%m-%d'),
                                    help="Choose one or more dates for intraday comparison analysis"
                                )
                            
                            with col2:
                                # Show number of data points for selected dates
                                if selected_dates:
                                    total_points = sum(len(df_processed[df_processed[timestamp_col].dt.date == date]) for date in selected_dates)
                                    st.metric("Total data points for selected dates", total_points)
                                else:
                                    st.metric("Total data points for selected dates", 0)
                            
                            if selected_dates:
                                # Filter data for selected dates
                                date_data = df_processed[df_processed[timestamp_col].dt.date.isin(selected_dates)]
                                
                                if len(date_data) > 0:
                                    st.markdown(f"### 📊 Intraday Analysis for {len(selected_dates)} Selected Day(s)")
                                    
                                    # Time Series for selected dates (overlaid)
                                    st.subheader("Overlaid Time Series Plot")
                                    
                                    # Create overlaid plot
                                    fig = go.Figure()
                                    
                                    for date in selected_dates:
                                        day_data = date_data[date_data[timestamp_col].dt.date == date]
                                        if len(day_data) > 0:
                                            for col in variable_cols:
                                                if col in day_data.columns:
                                                    # Create a DataFrame with timestamp and variable values
                                                    var_df = pd.DataFrame({
                                                        "timestamp": day_data[timestamp_col],
                                                        "value": day_data[col]
                                                    })
                                                    
                                                    # Remove any rows with NaN values
                                                    var_df = var_df.dropna()
                                                    
                                                    if len(var_df) > 0:
                                                        # Convert the actual timestamps to a common reference day
                                                        time_of_day_str = var_df['timestamp'].dt.strftime("%H:%M:%S")
                                                        common_day_times = pd.to_datetime(time_of_day_str, format="%H:%M:%S")
                                                        
                                                        if time_series_plot_type == "scatter":
                                                            fig.add_trace(
                                                                go.Scatter(
                                                                    x=common_day_times,
                                                                    y=var_df['value'],
                                                                    mode='markers',
                                                                    name=f"{col} - {date.strftime('%Y-%m-%d')}",
                                                                    marker=dict(size=4),
                                                                    hovertemplate=f'<b>{date.strftime("%Y-%m-%d")}</b><br>' +
                                                                                  f'Time: %{{x}}<br>' +
                                                                                  f'{col}: %{{y}}<extra></extra>'
                                                                )
                                                            )
                                                        else:  # line plot
                                                            fig.add_trace(
                                                                go.Scatter(
                                                                    x=common_day_times,
                                                                    y=var_df['value'],
                                                                    mode='lines',
                                                                    name=f"{col} - {date.strftime('%Y-%m-%d')}",
                                                                    line=dict(width=2),
                                                                    hovertemplate=f'<b>{date.strftime("%Y-%m-%d")}</b><br>' +
                                                                                  f'Time: %{{x}}<br>' +
                                                                                  f'{col}: %{{y}}<extra></extra>'
                                                                )
                                                            )
                                    
                                    fig.update_layout(
                                        title=f"Overlaid Intraday Time Series - {len(selected_dates)} Day(s)",
                                        xaxis_title="Time of Day",
                                        yaxis_title="Value",
                                        hovermode='x unified',
                                        height=500,
                                        showlegend=True,
                                        template="plotly_white"
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Day-specific correlation analysis (for first selected day)
                                    if len(variable_cols) >= 2 and len(selected_dates) == 1:
                                        st.subheader("Intraday Correlation Analysis")
                                        
                                        col1, col2 = st.columns([1, 1])
                                        with col1:
                                            show_day_correlation = st.checkbox(
                                                "Show intraday correlation",
                                                help="Display correlation between variables for the selected day",
                                                key="day_correlation_check"
                                            )
                                        
                                        with col2:
                                            if show_day_correlation:
                                                day_window_size = st.number_input(
                                                    "Correlation window size:",
                                                    min_value=2,
                                                    max_value=min(50, len(date_data) // 2),
                                                    value=min(10, len(date_data) // 4),
                                                    help="Number of data points for intraday correlation calculation",
                                                    key="window_size_day"
                                                )
                                        
                                        if show_day_correlation:
                                            # Calculate correlations for the day
                                            day_correlations = calculate_rolling_correlation(date_data, variable_cols, day_window_size)
                                            
                                            if day_correlations:
                                                day_correlation_fig = create_correlation_plot(
                                                    day_correlations, date_data, timestamp_col, correlation_plot_type,
                                                    f"Intraday Correlation - {selected_dates[0]} (Window: {day_window_size})"
                                                )
                                                st.plotly_chart(day_correlation_fig, use_container_width=True)
                                            else:
                                                st.warning("⚠️ Unable to calculate intraday correlations. Not enough data points.")
                                    
                                    # Multi-day correlation analysis (overlaid)
                                    if len(variable_cols) >= 2 and len(selected_dates) > 1:
                                        st.subheader("Multi-Day Correlation Analysis")
                                        
                                        col1, col2 = st.columns([1, 1])
                                        with col1:
                                            show_multi_day_correlation = st.checkbox(
                                                "Show multi-day correlation",
                                                help="Display correlation between variables over multiple days",
                                                key="multi_day_correlation_check"
                                            )
                                        
                                        with col2:
                                            if show_multi_day_correlation:
                                                multi_day_window_size = st.number_input(
                                                    "Correlation window size:",
                                                    min_value=2,
                                                    max_value=min(100, len(date_data) // 10),
                                                    value=min(20, len(date_data) // 20),
                                                    help="Number of data points for multi-day correlation calculation",
                                                    key="window_size_multi_day"
                                                )
                                        
                                        if show_multi_day_correlation:
                                            # Calculate correlations for all selected days
                                            multi_day_correlations = calculate_rolling_correlation(date_data, variable_cols, multi_day_window_size)
                                            
                                            if multi_day_correlations:
                                                # Create overlaid correlation plot normalized by time of day
                                                fig_corr = go.Figure()
                                                
                                                for corr_name, corr_values in multi_day_correlations.items():
                                                    if corr_values is not None and not corr_values.isna().all():
                                                        # Get the corresponding timestamp values
                                                        timestamp_values = date_data[timestamp_col].reindex(corr_values.index)
                                                        
                                                        # Create a DataFrame with timestamp and correlation values
                                                        corr_df = pd.DataFrame({
                                                            "timestamp": timestamp_values,
                                                            "rolling_corr": corr_values.values
                                                        })
                                                        
                                                        # Remove any rows with NaN values
                                                        corr_df = corr_df.dropna()
                                                        
                                                        if len(corr_df) > 0:
                                                            # Get unique days from the correlation data
                                                            unique_days = corr_df['timestamp'].dt.date.unique()
                                                            
                                                            for day in unique_days:
                                                                # Filter the data for the given day
                                                                day_mask = (corr_df['timestamp'].dt.date == day)
                                                                df_segment = corr_df.loc[day_mask].copy()
                                                                
                                                                if len(df_segment) > 0:
                                                                    # Convert the actual timestamps to a common reference day
                                                                    time_of_day_str = df_segment['timestamp'].dt.strftime("%H:%M:%S")
                                                                    common_day_times = pd.to_datetime(time_of_day_str, format="%H:%M:%S")
                                                                    
                                                                    if correlation_plot_type == "scatter":
                                                                        fig_corr.add_trace(
                                                                            go.Scatter(
                                                                                x=common_day_times,
                                                                                y=df_segment['rolling_corr'],
                                                                                mode='markers',
                                                                                name=f'{corr_name} - {day}',
                                                                                marker=dict(size=4),
                                                                                hovertemplate=f'{corr_name}<br>' +
                                                                                              f'Date: {day}<br>' +
                                                                                              f'Time: %{{x}}<br>' +
                                                                                              f'Correlation: %{{y:.3f}}<extra></extra>'
                                                                            )
                                                                        )
                                                                    else:  # line plot
                                                                        fig_corr.add_trace(
                                                                            go.Scatter(
                                                                                x=common_day_times,
                                                                                y=df_segment['rolling_corr'],
                                                                                mode='lines',
                                                                                name=f'{corr_name} - {day}',
                                                                                line=dict(width=2),
                                                                                hovertemplate=f'{corr_name}<br>' +
                                                                                              f'Date: {day}<br>' +
                                                                                              f'Time: %{{x}}<br>' +
                                                                                              f'Correlation: %{{y:.3f}}<extra></extra>'
                                                                            )
                                                                        )
                                                
                                                fig_corr.update_layout(
                                                    title=f"Multi-Day Correlation Analysis - Time of Day Overlay (Window: {multi_day_window_size})",
                                                    xaxis_title="Time of Day",
                                                    yaxis_title="Correlation Coefficient",
                                                    hovermode='x unified',
                                                    height=500,
                                                    showlegend=True,
                                                    template="plotly_white"
                                                )
                                                
                                                # Add horizontal line at y=0
                                                fig_corr.add_hline(y=0, line_dash="dash", line_color="gray")
                                                
                                                st.plotly_chart(fig_corr, use_container_width=True)
                                            else:
                                                st.warning("⚠️ Unable to calculate multi-day correlations. Check your data.")
                                    
                                    # Individual day statistics
                                    st.subheader("Individual Day Statistics")
                                    
                                    # Create tabs for each selected day
                                    if len(selected_dates) > 1:
                                        day_tabs = st.tabs([f"📊 {date.strftime('%Y-%m-%d')}" for date in selected_dates])
                                        
                                        for i, date in enumerate(selected_dates):
                                            with day_tabs[i]:
                                                day_data = date_data[date_data[timestamp_col].dt.date == date]
                                                if len(day_data) > 0:
                                                    st.markdown(f"**Summary Statistics for {date.strftime('%Y-%m-%d')}**")
                                                    day_stats = day_data[variable_cols].describe()
                                                    st.dataframe(day_stats, use_container_width=True)
                                                else:
                                                    st.warning(f"No data available for {date.strftime('%Y-%m-%d')}")
                                    else:
                                        # Single day - show statistics directly
                                        day_data = date_data[date_data[timestamp_col].dt.date == selected_dates[0]]
                                        if len(day_data) > 0:
                                            st.markdown(f"**Summary Statistics for {selected_dates[0].strftime('%Y-%m-%d')}**")
                                            day_stats = day_data[variable_cols].describe()
                                            st.dataframe(day_stats, use_container_width=True)
                                    
                                    # Multi-day statistics
                                    st.subheader("Combined Multi-Day Statistics")
                                    multi_day_stats = date_data[variable_cols].describe()
                                    st.dataframe(multi_day_stats, use_container_width=True)
                                    
                                    # Download multi-day data
                                    st.subheader("Export Multi-Day Data")
                                    multi_day_csv = date_data.to_csv(index=False)
                                    date_range_str = f"{selected_dates[0].strftime('%Y%m%d')}-{selected_dates[-1].strftime('%Y%m%d')}"
                                    st.download_button(
                                        label=f"📥 Download {len(selected_dates)} day(s) data as CSV",
                                        data=multi_day_csv,
                                        file_name=f"intraday_{date_range_str}_{uploaded_file.name}",
                                        mime="text/csv"
                                    )
                                    
                                else:
                                    st.warning(f"⚠️ No data available for selected dates")
                        else:
                            st.warning("⚠️ No valid dates found in the timestamp column")
                    else:
                        st.info("Please select timestamp and variable columns to view day-specific analysis.")
                
                with tab3:
                    # Hourly Bucketing Analysis
                    st.subheader("🕐 Hourly Bucketing Analysis")
                    st.markdown("Select an intraday time span to create 1-hour interval buckets with baseline statistics.")
                    
                    if timestamp_col and variable_cols:
                        # Time span selection
                        col1, col2 = st.columns(2)
                        with col1:
                            start_hour = st.selectbox(
                                "Start Hour (UTC):",
                                options=list(range(24)),
                                format_func=lambda x: f"{x:02d}:00",
                                help="Select the start hour for the intraday analysis"
                            )
                        
                        with col2:
                            end_hour = st.selectbox(
                                "End Hour (UTC):",
                                options=list(range(24)),
                                format_func=lambda x: f"{x:02d}:00",
                                index=10,  # Default to 10:00
                                help="Select the end hour for the intraday analysis"
                            )
                        
                        # Validate time span
                        if start_hour >= end_hour:
                            st.error("⚠️ End hour must be after start hour. Please adjust your selection.")
                        else:
                            # Calculate number of buckets
                            num_buckets = end_hour - start_hour
                            st.info(f"📊 Will create {num_buckets} hourly buckets from {start_hour:02d}:00 to {end_hour:02d}:00 UTC")
                            
                            # Filter data for the selected time span across all days
                            time_filtered_data = df_processed[
                                (df_processed[timestamp_col].dt.hour >= start_hour) &
                                (df_processed[timestamp_col].dt.hour < end_hour)
                            ].copy()
                            
                            if len(time_filtered_data) > 0:
                                # Add hour bucket column
                                time_filtered_data['hour_bucket'] = time_filtered_data[timestamp_col].dt.hour
                                
                                # Generate bucketing table
                                st.subheader("Hourly Bucketing Statistics")
                                
                                # Create comprehensive statistics for each hour bucket
                                bucket_stats = []
                                
                                for hour in range(start_hour, end_hour):
                                    hour_data = time_filtered_data[time_filtered_data['hour_bucket'] == hour]
                                    
                                    if len(hour_data) > 0:
                                        for col in variable_cols:
                                            if col in hour_data.columns:
                                                col_data = hour_data[col].dropna()
                                                if len(col_data) > 0:
                                                    bucket_stats.append({
                                                        'Hour': f"{hour:02d}:00-{(hour+1):02d}:00",
                                                        'Variable': col,
                                                        'Count': len(col_data),
                                                        'Mean': col_data.mean(),
                                                        'Std': col_data.std(),
                                                        'Min': col_data.min(),
                                                        'Max': col_data.max(),
                                                        'Median': col_data.median(),
                                                        'Q25': col_data.quantile(0.25),
                                                        'Q75': col_data.quantile(0.75),
                                                        'Range': col_data.max() - col_data.min(),
                                                        'Days_Available': hour_data[timestamp_col].dt.date.nunique()
                                                    })
                                
                                if bucket_stats:
                                    # Create DataFrame and display
                                    bucket_df = pd.DataFrame(bucket_stats)
                                    
                                    # Round numeric columns for better display
                                    numeric_cols = ['Mean', 'Std', 'Min', 'Max', 'Median', 'Q25', 'Q75', 'Range']
                                    bucket_df[numeric_cols] = bucket_df[numeric_cols].round(4)
                                    
                                    # Display the bucketing table
                                    st.dataframe(bucket_df, use_container_width=True)
                                    
                                    # Summary metrics
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Total Data Points", len(time_filtered_data))
                                    with col2:
                                        st.metric("Days Analyzed", time_filtered_data[timestamp_col].dt.date.nunique())
                                    with col3:
                                        st.metric("Hour Buckets", num_buckets)
                                    
                                    # Download bucketing data
                                    st.subheader("Export Bucketing Data")
                                    bucket_csv = bucket_df.to_csv(index=False)
                                    time_range_str = f"{start_hour:02d}00-{end_hour:02d}00"
                                    st.download_button(
                                        label=f"📥 Download Hourly Bucketing Data ({time_range_str})",
                                        data=bucket_csv,
                                        file_name=f"hourly_bucketing_{time_range_str}_{uploaded_file.name}",
                                        mime="text/csv"
                                    )
                                    
                                    # Optional: Show raw data for selected hour
                                    st.subheader("Raw Data by Hour")
                                    selected_hour = st.selectbox(
                                        "Select hour to view raw data:",
                                        options=[f"{h:02d}:00-{(h+1):02d}:00" for h in range(start_hour, end_hour)],
                                        help="View the actual data points for a specific hour bucket"
                                    )
                                    
                                    if selected_hour:
                                        hour_num = int(selected_hour.split(':')[0])
                                        hour_raw_data = time_filtered_data[time_filtered_data['hour_bucket'] == hour_num]
                                        
                                        if len(hour_raw_data) > 0:
                                            st.markdown(f"**Raw data for {selected_hour}:**")
                                            st.dataframe(hour_raw_data[[timestamp_col] + variable_cols].head(20), use_container_width=True)
                                            
                                            if len(hour_raw_data) > 20:
                                                st.info(f"Showing first 20 of {len(hour_raw_data)} data points. Use the download button above to get all data.")
                                        else:
                                            st.warning(f"No data available for {selected_hour}")
                                    
                                else:
                                    st.warning("⚠️ No data available for the selected time span and variables.")
                            else:
                                st.warning(f"⚠️ No data available between {start_hour:02d}:00 and {end_hour:02d}:00 UTC")
                    else:
                        st.info("Please select timestamp and variable columns to view hourly bucketing analysis.")
                
                # Statistics Section
                st.markdown('<h2 class="section-header">📊 Statistical Summary</h2>', unsafe_allow_html=True)
                
                if variable_cols:
                    # Descriptive statistics
                    st.subheader("Descriptive Statistics")
                    desc_stats = df_processed[variable_cols].describe()
                    st.dataframe(desc_stats, use_container_width=True)
                    
                    # Correlation matrix
                    if len(variable_cols) > 1:
                        st.subheader("Correlation Matrix")
                        corr_matrix = df_processed[variable_cols].corr()
                        
                        # Create heatmap
                        fig = px.imshow(
                            corr_matrix,
                            text_auto=True,
                            aspect="auto",
                            color_continuous_scale="RdBu",
                            title="Correlation Matrix Heatmap"
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Download processed data
                st.markdown('<h2 class="section-header">💾 Export Data</h2>', unsafe_allow_html=True)
                
                csv = df_processed.to_csv(index=False)
                st.download_button(
                    label="📥 Download processed data as CSV",
                    data=csv,
                    file_name=f"processed_{uploaded_file.name}",
                    mime="text/csv"
                )
    
    else:
        # Welcome message when no file is uploaded
        st.markdown("""
        <div class="info-box">
            <h3>🚀 Welcome to SeriesViz!</h3>
            <p>This dashboard helps you visualize and analyze time series data. Here's what you can do:</p>
            <ul>
                <li>📁 Upload CSV files with time series data</li>
                <li>⏰ Select timestamp and variable columns</li>
                <li>🔧 Clean data by removing NaN values</li>
                <li>📊 Normalize data using Z-score</li>
                <li>📈 Create interactive time series plots</li>
                <li>🔄 Analyze rolling correlations between variables</li>
                <li>📋 View statistical summaries and correlation matrices</li>
                <li>💾 Export processed data</li>
            </ul>
            <p><strong>To get started:</strong> Upload a CSV file using the sidebar on the left.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Example data structure
        st.subheader("📋 Expected Data Structure")
        st.markdown("""
        Your CSV file should have:
        - **One timestamp column**: Contains date/time information (e.g., '2023-01-01', '2023-01-01 10:30:00')
        - **One or more variable columns**: Contains numerical data to analyze
        - **Consistent data types**: Timestamp column should be parseable as dates
        """)
        
        # Sample data
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='D'),
            'temperature': [20.1, 22.3, 19.8, 25.2, 23.1, 18.9, 21.5, 24.7, 20.3, 22.8],
            'humidity': [65, 58, 72, 45, 52, 78, 63, 48, 69, 55],
            'pressure': [1013.2, 1010.8, 1015.6, 1008.9, 1012.3, 1017.1, 1011.5, 1009.7, 1014.2, 1010.1]
        })
        
        st.subheader("📄 Sample Data Format")
        st.dataframe(sample_data, use_container_width=True)

if __name__ == "__main__":
    main() 