# SeriesViz: Time Series Financial Analysis Toolkit

A professional toolkit for visualizing and analyzing time series data, specifically designed for financial applications. This Streamlit-based dashboard enables quantitative researchers and analysts to upload, process, and explore complex datasets to generate hypotheses and identify potential signals.

## Core Features

- **Data Ingestion**: Upload CSV files directly into the dashboard for immediate analysis.
- **Timestamp Processing**: Automatically detects and converts timestamp columns to a standardized datetime format.
- **Variable Selection**: Isolate specific data columns for focused analysis and visualization.
- **Data Preprocessing**: Includes options for handling missing values (NaN removal) and data normalization (Z-score).
- **Interactive Visualization**: Generate interactive time series plots using Plotly for in-depth exploration of temporal data.
- **Quantitative Analysis Tools**:
    - **Rolling Correlation Analysis**: Analyze the correlation between multiple time series over user-defined rolling windows. This is crucial for understanding dynamic relationships between financial instruments or economic indicators.
    - **Statistical Summaries**: Generate descriptive statistics and correlation matrices to provide a quantitative overview of the dataset.
- **Hypothesis Generation**: The visualization and analysis tools are designed to help researchers formulate and test hypotheses about market behavior and relationships between financial assets.
- **Signal Exploration**: By analyzing correlations and trends, users can explore potential alpha signals for trading strategies.
- **Data Export**: Download the processed and analyzed data as a CSV file for further offline analysis or backtesting.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup Instructions

1.  **Clone the repository**
    ```bash
    git clone <repository-url>
    cd SeriesViz
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the dashboard**
    ```bash
    ./run_dashboard.sh
    ```

4.  **Access the application**
    - The dashboard will be available at `http://localhost:8501`.

## Usage Guide

1.  **Upload Data**: Use the file uploader in the sidebar to load a CSV file.
2.  **Configure Data**:
    - **Timestamp Column**: Select the column containing the timestamp information.
    - **Variable Columns**: Select the numerical columns for analysis.
3.  **Preprocess Data**:
    - **Remove NaN Values**: Enable this option to clean the dataset.
    - **Z-score Normalization**: Standardize variables for comparative analysis.
4.  **Analyze**:
    - **Time Series Plot**: Visualize the selected variables over time.
    - **Rolling Correlation**: Set a window size and analyze the dynamic correlation between variables.
    - **Statistical Summary**: Review summary statistics and correlation matrices.
5.  **Export**: Download the processed data for use in other tools or models.

## Data Requirements

### CSV File Structure
- The CSV file must contain at least one timestamp column and one or more numerical data columns.
- The timestamp column should be in a format that is parsable by pandas (e.g., 'YYYY-MM-DD', 'YYYY-MM-DD HH:MM:SS').

### Example Data Format
```csv
timestamp,asset_price_1,asset_price_2,volume
2023-01-01,150.1,200.5,10000
2023-01-02,152.3,201.2,12000
2023-01-03,151.8,203.6,11500
...
```

## Technical Details

### Dependencies
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **plotly**: Interactive plotting library
- **numpy**: Numerical computing
- **scipy**: Scientific computing (for Z-score normalization)

### Architecture
The application is designed with a modular structure to ensure clarity and extensibility. Key functions are separated by responsibility, such as data loading, processing, and visualization, which allows for easier maintenance and addition of new analysis modules.
