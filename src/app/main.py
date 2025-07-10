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
from src.analysis.correlation import calculate_bucket_correlation_metrics, calculate_weighted_correlation, calculate_rolling_correlation
from src.analysis.spread import calculate_spread_metrics, calculate_bucket_spread_metrics
from src.analysis.raw_metrics import calculate_bucket_raw_metrics
from src.analysis.raw_metrics import calculate_bucket_raw_metrics
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

def create_correlation_plot(correlations, df, timestamp_col, plot_type="line", title="Rolling Correlation", use_r_squared=False):
    """Create correlation plot using Plotly"""
    if not correlations:
        return None
    
    # Update title and axis labels based on R² toggle
    if use_r_squared:
        title = title.replace("Correlation", "R²")
        y_axis_title = "R² Value"
    else:
        y_axis_title = "Correlation Coefficient"
    
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
        yaxis_title=y_axis_title,
        hovermode='x unified',
        height=500,
        showlegend=True,
        template="plotly_white"
    )
    
    # Add horizontal line at y=0 for correlation, or at y=0.5 for R² (typical threshold)
    if use_r_squared:
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="R² = 0.5")
    else:
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
        
        # Analysis Settings
        st.markdown('<h3 class="section-header">⚙️ Analysis Settings</h3>', unsafe_allow_html=True)
        
        # Analysis type selection
        analysis_type = st.radio(
            "Analysis Type:",
            options=["Correlation", "R²", "Spread Analysis"],
            help="Choose the type of analysis to perform: Correlation (r), R² (r²), or Spread Analysis (variability metrics)"
        )
        
        # Set flags based on selection
        use_r_squared = (analysis_type == "R²")
        use_spread_analysis = (analysis_type == "Spread Analysis")
        
        # Weighted correlation settings (only for correlation analysis)
        if analysis_type == "Correlation":
            st.markdown('<h4 class="section-header">🔧 Correlation Settings</h4>', unsafe_allow_html=True)
            use_weighted_correlation = st.checkbox(
                "Use Weighted Correlation",
                help="Weight correlation by magnitude of movements (large moves have more impact)"
            )
            
            if use_weighted_correlation:
                weight_type = st.selectbox(
                    "Weighting Method:",
                    options=["magnitude", "change_magnitude", "volatility"],
                    format_func=lambda x: {
                        "magnitude": "Absolute Values (|x| + |y|)",
                        "change_magnitude": "Change Magnitude (|Δx| + |Δy|)",
                        "volatility": "Volatility (√(σ²x + σ²y))"
                    }[x],
                    help="Choose how to weight the correlation calculation"
                )
        else:
            use_weighted_correlation = False
            weight_type = "magnitude"
        
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
                            f"Show rolling {'spread analysis' if use_spread_analysis else 'correlation'}",
                            help=f"Display {'spread metrics' if use_spread_analysis else 'correlation'} between variables over time"
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
                        if use_spread_analysis:
                            # Calculate spread metrics
                            spread_metrics = calculate_spread_metrics(df_processed, variable_cols)
                            
                            if spread_metrics:
                                # Create spread analysis plot
                                fig_spread = go.Figure()
                                
                                # Get available spread metrics for selection
                                available_metrics = list(spread_metrics[list(spread_metrics.keys())[0]].keys()) if spread_metrics else []
                                
                                # Add metric selection
                                spread_metric = st.selectbox(
                                    "Spread metric to display:",
                                    options=available_metrics,
                                    help="Choose which spread metric to visualize",
                                    key="spread_metric_select"
                                )
                                
                                for pair_name, pair_metrics in spread_metrics.items():
                                    if spread_metric in pair_metrics:
                                        metric_values = pair_metrics[spread_metric]
                                        if not metric_values.isna().all():
                                            # Get the corresponding timestamps
                                            timestamp_values = df_processed[timestamp_col].reindex(metric_values.index)
                                            
                                            fig_spread.add_trace(
                                                go.Scatter(
                                                    x=timestamp_values,
                                                    y=metric_values,
                                                    mode='lines',
                                                    name=pair_name,
                                                    line=dict(width=2)
                                                )
                                            )
                                
                                fig_spread.update_layout(
                                    title=f"Spread Analysis - {spread_metric}",
                                    xaxis_title="Time",
                                    yaxis_title=f"{spread_metric} Value",
                                    hovermode='x unified',
                                    height=500,
                                    showlegend=True,
                                    template="plotly_white"
                                )
                                
                                st.plotly_chart(fig_spread, use_container_width=True)
                                
                                # Spread metrics summary
                                st.subheader("Spread Metrics Summary")
                                spread_summary = {}
                                for pair_name, pair_metrics in spread_metrics.items():
                                    pair_summary = {}
                                    for metric in available_metrics:
                                        if metric in pair_metrics:
                                            values = pair_metrics[metric].dropna()
                                            if len(values) > 0:
                                                pair_summary[metric] = {
                                                    'Mean': values.mean(),
                                                    'Std': values.std(),
                                                    'Min': values.min(),
                                                    'Max': values.max()
                                                }
                                    spread_summary[pair_name] = pair_summary
                                
                                # Create summary DataFrame
                                summary_data = []
                                for pair_name, metrics in spread_summary.items():
                                    for metric, stats in metrics.items():
                                        summary_data.append({
                                            'Variable Pair': pair_name,
                                            'Metric': metric,
                                            'Mean': round(stats['Mean'], 4),
                                            'Std': round(stats['Std'], 4),
                                            'Min': round(stats['Min'], 4),
                                            'Max': round(stats['Max'], 4)
                                        })
                                
                                if summary_data:
                                    summary_df = pd.DataFrame(summary_data)
                                    st.dataframe(summary_df, use_container_width=True)
                            else:
                                st.warning("⚠️ Unable to calculate spread metrics. Check your data.")
                        else:
                            # Calculate correlations
                            if use_weighted_correlation:
                                correlations = calculate_weighted_correlation(df_processed, variable_cols, window_size, weight_type)
                            else:
                                correlations = calculate_rolling_correlation(df_processed, variable_cols, window_size, use_r_squared)
                            
                            if correlations:
                                correlation_fig = create_correlation_plot(
                                    correlations, df_processed, timestamp_col, correlation_plot_type,
                                    f"Rolling {'Weighted ' if use_weighted_correlation else ''}{'R²' if use_r_squared else 'Correlation'} (Window Size: {window_size})", use_r_squared
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
                                    
                                    # Plot type selection for day-specific analysis
                                    day_plot_type = st.radio(
                                        "Day-Specific Plot Type:",
                                        options=["scatter", "lines"],
                                        horizontal=True,
                                        help="Choose between scatter plots (individual points) or line plots (connected points)"
                                    )
                                    
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
                                                        
                                                        # Use selected plot type for day-specific analysis
                                                        if day_plot_type == "scatter":
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
                                                        else:  # lines plot
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
                                                f"Show intraday {'spread analysis' if use_spread_analysis else 'correlation'}",
                                                help=f"Display {'spread metrics' if use_spread_analysis else 'correlation'} between variables for the selected day",
                                                key="day_correlation_check"
                                            )
                                        
                                        with col2:
                                            if show_day_correlation and not use_spread_analysis:
                                                day_window_size = st.number_input(
                                                    f"{'Correlation'} window size:",
                                                    min_value=2,
                                                    max_value=min(50, len(date_data) // 2),
                                                    value=min(10, len(date_data) // 4),
                                                    help=f"Number of data points for intraday {'correlation'} calculation",
                                                    key="window_size_day"
                                                )
                                        
                                        if show_day_correlation:
                                            if use_spread_analysis:
                                                # Calculate spread metrics for the day
                                                day_spread_metrics = calculate_spread_metrics(date_data, variable_cols)
                                                
                                                if day_spread_metrics:
                                                    # Create spread analysis plot for the day
                                                    fig_spread_day = go.Figure()
                                                    
                                                    # Get available spread metrics for selection
                                                    available_metrics = list(day_spread_metrics[list(day_spread_metrics.keys())[0]].keys()) if day_spread_metrics else []
                                                    
                                                    # Add metric selection for day analysis
                                                    day_spread_metric = st.selectbox(
                                                        "Spread metric to display:",
                                                        options=available_metrics,
                                                        help="Choose which spread metric to visualize",
                                                        key="day_spread_metric_select"
                                                    )
                                                    
                                                    for pair_name, pair_metrics in day_spread_metrics.items():
                                                        if day_spread_metric in pair_metrics:
                                                            metric_values = pair_metrics[day_spread_metric]
                                                            if not metric_values.isna().all():
                                                                # Get the corresponding timestamps
                                                                timestamp_values = date_data[timestamp_col].reindex(metric_values.index)
                                                                
                                                                fig_spread_day.add_trace(
                                                                    go.Scatter(
                                                                        x=timestamp_values,
                                                                        y=metric_values,
                                                                        mode='markers' if day_plot_type == "scatter" else 'lines',
                                                                        name=pair_name,
                                                                        marker=dict(size=4) if day_plot_type == "scatter" else None,
                                                                        line=dict(width=2) if day_plot_type == "lines" else None
                                                                    )
                                                                )
                                                    
                                                    fig_spread_day.update_layout(
                                                        title=f"Intraday {day_spread_metric} Analysis - {selected_dates[0]}",
                                                        xaxis_title="Time",
                                                        yaxis_title=f"{day_spread_metric} Value",
                                                        hovermode='x unified',
                                                        height=500,
                                                        showlegend=True,
                                                        template="plotly_white"
                                                    )
                                                    
                                                    st.plotly_chart(fig_spread_day, use_container_width=True)
                                                else:
                                                    st.warning("⚠️ Unable to calculate intraday spread metrics. Not enough data points.")
                                            else:
                                                # Calculate correlations for the day
                                                if use_weighted_correlation:
                                                    day_correlations = calculate_weighted_correlation(date_data, variable_cols, day_window_size, weight_type)
                                                else:
                                                    day_correlations = calculate_rolling_correlation(date_data, variable_cols, day_window_size, use_r_squared)
                                                
                                                if day_correlations:
                                                    day_correlation_fig = create_correlation_plot(
                                                        day_correlations, date_data, timestamp_col, day_plot_type,
                                                        f"Intraday {'Weighted ' if use_weighted_correlation else ''}{'R²' if use_r_squared else 'Correlation'} - {selected_dates[0]} (Window: {day_window_size})", use_r_squared
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
                                                f"Show multi-day {'spread analysis' if use_spread_analysis else 'correlation'}",
                                                help=f"Display {'spread metrics' if use_spread_analysis else 'correlation'} between variables over multiple days",
                                                key="multi_day_correlation_check"
                                            )
                                        
                                        with col2:
                                            if show_multi_day_correlation and not use_spread_analysis:
                                                multi_day_window_size = st.number_input(
                                                    f"{'Correlation'} window size:",
                                                    min_value=2,
                                                    max_value=min(100, len(date_data) // 10),
                                                    value=min(20, len(date_data) // 20),
                                                    help=f"Number of data points for multi-day {'correlation'} calculation",
                                                    key="window_size_multi_day"
                                                )
                                        
                                        if show_multi_day_correlation:
                                            if use_spread_analysis:
                                                # Calculate spread metrics for all selected days
                                                multi_day_spread_metrics = calculate_spread_metrics(date_data, variable_cols)
                                                
                                                if multi_day_spread_metrics:
                                                    # Create overlaid spread analysis plot normalized by time of day
                                                    fig_spread_multi = go.Figure()
                                                    
                                                    # Get available spread metrics for selection
                                                    available_metrics = list(multi_day_spread_metrics[list(multi_day_spread_metrics.keys())[0]].keys()) if multi_day_spread_metrics else []
                                                    
                                                    # Add metric selection for multi-day analysis
                                                    multi_day_spread_metric = st.selectbox(
                                                        "Spread metric to display:",
                                                        options=available_metrics,
                                                        help="Choose which spread metric to visualize",
                                                        key="multi_day_spread_metric_select"
                                                    )
                                                    
                                                    for pair_name, pair_metrics in multi_day_spread_metrics.items():
                                                        if multi_day_spread_metric in pair_metrics:
                                                            # Get metric values
                                                            metric_values = pair_metrics[multi_day_spread_metric]
                                                            if not metric_values.isna().all():
                                                                # Create a DataFrame with timestamp and metric values
                                                                metric_df = pd.DataFrame({
                                                                    "timestamp": date_data[timestamp_col],
                                                                    "metric": metric_values.values
                                                                })
                                                                
                                                                # Remove any rows with NaN values
                                                                metric_df = metric_df.dropna()
                                                                
                                                                if len(metric_df) > 0:
                                                                    # Get unique days from the metric data
                                                                    unique_days = metric_df['timestamp'].dt.date.unique()
                                                                    first_trace = True
                                                                    for day in unique_days:
                                                                        # Filter the data for the given day
                                                                        day_mask = (metric_df['timestamp'].dt.date == day)
                                                                        df_segment = metric_df.loc[day_mask].copy()
                                                                        
                                                                        if len(df_segment) > 0:
                                                                            # Convert the actual timestamps to a common reference day
                                                                            time_of_day_str = df_segment['timestamp'].dt.strftime("%H:%M:%S")
                                                                            common_day_times = pd.to_datetime(time_of_day_str, format="%H:%M:%S")
                                                                            
                                                                            fig_spread_multi.add_trace(
                                                                                go.Scatter(
                                                                                    x=common_day_times,
                                                                                    y=df_segment['metric'],
                                                                                    mode='markers' if day_plot_type == "scatter" else 'lines',
                                                                                    name=f"{pair_name}" if first_trace else None,
                                                                                    showlegend=first_trace,
                                                                                    marker=dict(size=4) if day_plot_type == "scatter" else None,
                                                                                    line=dict(width=2) if day_plot_type == "lines" else None,
                                                                                    hovertemplate=f'{pair_name}<br>' +
                                                                                                  f'Date: {day}<br>' +
                                                                                                  f'Time: %{{x}}<br>' +
                                                                                                  f'{multi_day_spread_metric}: %{{y:.3f}}<extra></extra>'
                                                                                )
                                                                            )
                                                                            first_trace = False
                                                    
                                                    fig_spread_multi.update_layout(
                                                        title=f"Multi-Day {multi_day_spread_metric} Analysis - Time of Day Overlay",
                                                        xaxis_title="Time of Day",
                                                        yaxis_title=f"{multi_day_spread_metric} Value",
                                                        hovermode='x unified',
                                                        height=500,
                                                        showlegend=True,
                                                        template="plotly_white"
                                                    )
                                                    
                                                    st.plotly_chart(fig_spread_multi, use_container_width=True)
                                                else:
                                                    st.warning("⚠️ Unable to calculate multi-day spread metrics. Check your data.")
                                            else:
                                                # Calculate correlations for all selected days
                                                if use_weighted_correlation:
                                                    multi_day_correlations = calculate_weighted_correlation(date_data, variable_cols, multi_day_window_size, weight_type)
                                                else:
                                                    multi_day_correlations = calculate_rolling_correlation(date_data, variable_cols, multi_day_window_size, use_r_squared)
                                                
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
                                                                first_trace = True
                                                                for day in unique_days:
                                                                    # Filter the data for the given day
                                                                    day_mask = (corr_df['timestamp'].dt.date == day)
                                                                    df_segment = corr_df.loc[day_mask].copy()
                                                                    
                                                                    if len(df_segment) > 0:
                                                                        # Convert the actual timestamps to a common reference day
                                                                        time_of_day_str = df_segment['timestamp'].dt.strftime("%H:%M:%S")
                                                                        common_day_times = pd.to_datetime(time_of_day_str, format="%H:%M:%S")
                                                                        
                                                                        # Use selected plot type for multi-day analysis
                                                                        if day_plot_type == "scatter":
                                                                            fig_corr.add_trace(
                                                                                go.Scatter(
                                                                                    x=common_day_times,
                                                                                    y=df_segment['rolling_corr'],
                                                                                    mode='markers',
                                                                                    name=corr_name if first_trace else None,
                                                                                    showlegend=first_trace,
                                                                                    marker=dict(size=4),
                                                                                    hovertemplate=f'{corr_name}<br>' +
                                                                                                  f'Date: {day}<br>' +
                                                                                                  f'Time: %{{x}}<br>' +
                                                                                                  f'{"R²" if use_r_squared else "Correlation"}: %{{y:.3f}}<extra></extra>'
                                                                                )
                                                                            )
                                                                        else:  # lines plot
                                                                            fig_corr.add_trace(
                                                                                go.Scatter(
                                                                                    x=common_day_times,
                                                                                    y=df_segment['rolling_corr'],
                                                                                    mode='lines',
                                                                                    name=corr_name if first_trace else None,
                                                                                    showlegend=first_trace,
                                                                                    line=dict(width=2),
                                                                                    hovertemplate=f'{corr_name}<br>' +
                                                                                                  f'Time: %{{x}}<br>' +
                                                                                                  f'{"R²" if use_r_squared else "Correlation"}: %{{y:.3f}}<extra></extra>'
                                                                                )
                                                                            )
                                                                        first_trace = False
                                                    
                                                    fig_corr.update_layout(
                                                        title=f"Multi-Day {'Weighted ' if use_weighted_correlation else ''}{'R²' if use_r_squared else 'Correlation'} Analysis - Time of Day Overlay (Window: {multi_day_window_size})",
                                                        xaxis_title="Time of Day",
                                                        yaxis_title="R² Value" if use_r_squared else "Correlation Coefficient",
                                                        hovermode='x unified',
                                                        height=500,
                                                        showlegend=True,
                                                        template="plotly_white"
                                                    )
                                                    
                                                    # Add horizontal line at y=0 for correlation, or at y=0.5 for R²
                                                    if use_r_squared:
                                                        fig_corr.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="R² = 0.5")
                                                    else:
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
                    st.markdown("Select an intraday time span to create 1-hour interval buckets with baseline statistics and correlations.")
                    
                    if timestamp_col and variable_cols:
                        # Time span and bucket size selection
                        col1, col2, col3 = st.columns(3)
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
                        with col3:
                            bucket_minutes = st.number_input(
                                "Bucket Size (minutes):",
                                min_value=1,
                                max_value=120,
                                value=60,
                                step=1,
                                help="Set the size of each time bucket in minutes (e.g., 10, 15, 30, 60)"
                            )
                        
                        # Validate time span
                        if start_hour >= end_hour:
                            st.error("⚠️ End hour must be after start hour. Please adjust your selection.")
                        else:
                            # Calculate bucket edges
                            start_time = pd.Timestamp(f"2000-01-01 {start_hour:02d}:00:00")
                            end_time = pd.Timestamp(f"2000-01-01 {end_hour:02d}:00:00")
                            bucket_edges = pd.date_range(start=start_time, end=end_time, freq=f'{bucket_minutes}min')
                            if bucket_edges[-1] > end_time:
                                bucket_edges = bucket_edges[:-1]
                            num_buckets = len(bucket_edges) - 1
                            st.info(f"📊 Will create {num_buckets} buckets of {bucket_minutes} minutes from {start_hour:02d}:00 to {end_hour:02d}:00 UTC")
                            
                            # Filter data for the selected time span across all days
                            time_filtered_data = df_processed[
                                (df_processed[timestamp_col].dt.hour >= start_hour) &
                                (df_processed[timestamp_col].dt.hour < end_hour)
                            ].copy()
                            
                            if len(time_filtered_data) > 0:
                                # Assign each row to a bucket
                                time_of_day = time_filtered_data[timestamp_col].dt.time
                                minutes_since_midnight = time_filtered_data[timestamp_col].dt.hour * 60 + time_filtered_data[timestamp_col].dt.minute
                                bucket_labels = []
                                for i in range(num_buckets):
                                    left = bucket_edges[i].hour * 60 + bucket_edges[i].minute
                                    right = bucket_edges[i+1].hour * 60 + bucket_edges[i+1].minute
                                    bucket_labels.append(f"{bucket_edges[i].strftime('%H:%M')}-{bucket_edges[i+1].strftime('%H:%M')}")
                                time_filtered_data['bucket_idx'] = pd.cut(
                                    minutes_since_midnight,
                                    bins=[bucket_edges[i].hour * 60 + bucket_edges[i].minute for i in range(len(bucket_edges))],
                                    labels=bucket_labels,
                                    include_lowest=True,
                                    right=False
                                )
                                # Only keep rows with a valid bucket
                                time_filtered_data = time_filtered_data[~time_filtered_data['bucket_idx'].isna()]
                                
                                # Debug: Check bucket_labels
                                st.write(f"Debug: bucket_labels type: {type(bucket_labels)}")
                                st.write(f"Debug: bucket_labels content: {bucket_labels}")
                                st.write(f"Debug: bucket_labels length: {len(bucket_labels)}")
                                
                                # --- Raw Metrics Table ---
                                st.subheader(f"Raw Metrics Table (Variables){' - Including Spread Analysis' if use_spread_analysis else ''}")
                                raw_metrics_df = calculate_bucket_raw_metrics(
                                    time_filtered_data, variable_cols, bucket_labels, 'bucket_idx', use_spread_analysis)
                                st.dataframe(raw_metrics_df, use_container_width=True)
                                st.download_button(
                                    label=f"📥 Download Raw Metrics Table (CSV){' - With Spread' if use_spread_analysis else ''}",
                                    data=raw_metrics_df.to_csv(),
                                    file_name=f"raw_metrics_{'spread_' if use_spread_analysis else ''}{bucket_minutes}min_{start_hour:02d}-{end_hour:02d}.csv",
                                    mime="text/csv"
                                )
                                
                                # --- Correlation Metrics Table ---
                                if use_spread_analysis:
                                    st.subheader("Spread Metrics Table (Variable Pairs)")
                                    # For spread analysis, we'll show spread metrics for each variable pair in each bucket
                                    spread_metrics_df = calculate_bucket_spread_metrics(
                                        time_filtered_data, variable_cols, bucket_labels, 'bucket_idx')
                                    st.dataframe(spread_metrics_df, use_container_width=True)
                                    st.download_button(
                                        label="📥 Download Spread Metrics Table (CSV)",
                                        data=spread_metrics_df.to_csv(),
                                        file_name=f"spread_metrics_{bucket_minutes}min_{start_hour:02d}-{end_hour:02d}.csv",
                                        mime="text/csv"
                                    )
                                else:
                                    st.subheader(f"{'R²' if use_r_squared else 'Correlation'} Metrics Table (Variable Pairs)")
                                    corr_window_size = st.number_input(
                                        "Rolling window size for correlation:",
                                        min_value=2,
                                        max_value=50,
                                        value=5,
                                        help="Window size for rolling correlation in each bucket",
                                        key="corr_metrics_window_size"
                                    )
                                    corr_metrics_df, debug_hour_counts = calculate_bucket_correlation_metrics(
                                        time_filtered_data, variable_cols, bucket_labels, 'bucket_idx', corr_window_size, use_r_squared
                                    )
                                    st.dataframe(corr_metrics_df, use_container_width=True)
                                    st.download_button(
                                        label=f"📥 Download {'R²' if use_r_squared else 'Correlation'} Metrics Table (CSV)",
                                        data=corr_metrics_df.to_csv(),
                                        file_name=f"{'r2' if use_r_squared else 'corr'}_metrics_{bucket_minutes}min_{start_hour:02d}-{end_hour:02d}.csv",
                                        mime="text/csv"
                                    )
                                    # st.info(f"Data points per bucket for correlation (var1, var2): {debug_hour_counts}")
                                
                                # Summary metrics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Data Points", len(time_filtered_data))
                                with col2:
                                    st.metric("Days Analyzed", time_filtered_data[timestamp_col].dt.date.nunique())
                                with col3:
                                    st.metric("Buckets", num_buckets)
                                
                                # Optional: Show raw data for selected bucket
                                st.subheader("Raw Data by Bucket")
                                selected_bucket = st.selectbox(
                                    "Select bucket to view raw data:",
                                    options=bucket_labels,
                                    help="View the actual data points for a specific time bucket"
                                )
                                
                                if selected_bucket:
                                    bucket_raw_data = time_filtered_data[time_filtered_data['bucket_idx'] == selected_bucket]
                                    
                                    if len(bucket_raw_data) > 0:
                                        st.markdown(f"**Raw data for {selected_bucket}:**")
                                        st.dataframe(bucket_raw_data[[timestamp_col] + variable_cols].head(20), use_container_width=True)
                                        
                                        if len(bucket_raw_data) > 20:
                                            st.info(f"Showing first 20 of {len(bucket_raw_data)} data points. Use the download button above to get all data.")
                                    else:
                                        st.warning(f"No data available for {selected_bucket}")
                                
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
                <li> Create interactive time series plots</li>
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