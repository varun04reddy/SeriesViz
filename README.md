# 📊 SeriesViz - Time Series Data Visualization Dashboard

A powerful and intuitive Streamlit dashboard for visualizing and analyzing time series data. Upload your CSV files, configure your data, and explore interactive visualizations with ease.

## 🚀 Features

### Core Functionality
- **📁 File Upload**: Drag and drop CSV files directly into the dashboard
- **⏰ Timestamp Selection**: Automatically detect and convert timestamp columns to datetime format
- **📊 Variable Selection**: Choose which columns to analyze and visualize
- **🔧 Data Preprocessing**: Remove NaN values and apply Z-score normalization
- **📈 Interactive Visualizations**: Create beautiful time series plots using Plotly
- **🔄 Rolling Correlation Analysis**: Analyze correlations between variables over time with configurable window sizes
- **📋 Statistical Summaries**: View descriptive statistics and correlation matrices
- **💾 Data Export**: Download processed data for further analysis

### User Experience
- **🎨 Modern UI**: Clean, intuitive interface with responsive design
- **📱 Responsive Layout**: Works seamlessly on desktop and mobile devices
- **⚡ Real-time Updates**: Instant visualization updates as you modify settings
- **🛡️ Error Handling**: Robust error handling with helpful error messages
- **📖 Helpful Tooltips**: Contextual help throughout the interface

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup Instructions

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd SeriesViz
   ```

2. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   - The dashboard will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, navigate to the URL manually

## 📖 Usage Guide

### 1. Upload Your Data
- Click "Browse files" in the sidebar or drag and drop your CSV file
- Supported format: CSV files only
- The dashboard will automatically load and display your data

### 2. Configure Your Data
- **Timestamp Column**: Select the column containing your time/date information
  - The dashboard will automatically convert it to datetime format
  - Supported formats: '2023-01-01', '2023-01-01 10:30:00', etc.
- **Variable Columns**: Select one or more columns to analyze
  - These should contain numerical data you want to visualize

### 3. Preprocess Your Data
- **Remove NaN Values**: Check this option to remove rows with missing values
- **Z-score Normalization**: Check this option to standardize your variables
  - This is useful when variables have different scales

### 4. Explore Visualizations
- **Time Series Plot**: Interactive plot showing your variables over time
- **Rolling Correlation**: Analyze how correlations between variables change over time
  - Adjust the window size to control the correlation calculation period
- **Statistical Summary**: View descriptive statistics and correlation matrices

### 5. Export Your Results
- Download the processed data as a CSV file for further analysis

## 📋 Data Requirements

### Expected CSV Structure
Your CSV file should have:
- **One timestamp column**: Contains date/time information
- **One or more variable columns**: Contains numerical data to analyze
- **Consistent data types**: Timestamp column should be parseable as dates

### Example Data Format
```csv
timestamp,temperature,humidity,pressure
2023-01-01,20.1,65,1013.2
2023-01-02,22.3,58,1010.8
2023-01-03,19.8,72,1015.6
...
```

## 🔧 Technical Details

### Dependencies
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **plotly**: Interactive plotting library
- **numpy**: Numerical computing
- **scipy**: Scientific computing (for Z-score normalization)

### Architecture
The dashboard follows SOLID principles and KISS design:
- **Single Responsibility**: Each function has one clear purpose
- **Open/Closed**: Easy to extend with new features
- **Dependency Inversion**: Modular design with clear interfaces
- **Keep It Simple**: Intuitive user interface and straightforward code
- **You Aren't Gonna Need It**: Only implements currently needed features

### Key Functions
- `load_data()`: Handles CSV file loading with error handling
- `convert_to_datetime()`: Converts timestamp columns to datetime format
- `remove_nan_values()`: Cleans data by removing missing values
- `z_score_normalize()`: Standardizes variables using Z-score
- `calculate_rolling_correlation()`: Computes rolling correlations between variables
- `create_time_series_plot()`: Generates interactive time series visualizations
- `create_correlation_plot()`: Creates rolling correlation plots

## 🎯 Use Cases

### Ideal For
- **IoT Data Analysis**: Sensor data, device metrics, environmental monitoring
- **Financial Time Series**: Stock prices, market indicators, trading data
- **Scientific Research**: Experimental data, climate data, medical measurements
- **Business Analytics**: Sales data, user metrics, performance indicators
- **Quality Control**: Manufacturing data, process monitoring

### Example Scenarios
1. **Weather Station Data**: Analyze temperature, humidity, and pressure correlations
2. **Stock Market Analysis**: Compare multiple stock prices over time
3. **Sensor Network**: Monitor multiple sensors in a manufacturing facility
4. **Website Analytics**: Track user engagement metrics over time

## 🚀 Getting Started Quickly

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run the app**: `streamlit run app.py`
3. **Upload a CSV file** with timestamp and numerical columns
4. **Select your timestamp and variable columns**
5. **Explore the visualizations!**

## 🤝 Contributing

This project follows clean code principles and welcomes contributions:
- Follow SOLID principles
- Keep functions simple and focused
- Add comprehensive error handling
- Include helpful documentation

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

If you encounter any issues:
1. Check that your CSV file has the correct format
2. Ensure your timestamp column contains parseable dates
3. Verify that variable columns contain numerical data
4. Check the console for any error messages

---

**Happy visualizing! 📊✨**