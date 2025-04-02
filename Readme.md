# Iowa Liquor Sales Dashboard

An interactive Streamlit dashboard analyzing Iowa liquor sales data, featuring county-level analysis, sales comparisons, and forecasting capabilities.

## Features

- **Personal Bio**: Introduction and project overview
- **County Analysis**: 
  - Monthly sales trends by county
  - Top 10 performing stores
  - Top selling categories
  - Interactive county map with category visualization
- **Sales Comparisons**:
  - Category performance analysis
  - Store performance metrics
  - Monthly sales and volume trends
- **Forecasting**:
  - Category-level sales forecasting
  - Seasonal pattern analysis
  - Item-level predictions

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/joshkozak/Iowa-Liquor-Dashboard.git
cd Iowa-Liquor-Dashboard
```

2. Install required packages:
```bash
pip install streamlit pandas numpy plotly folium streamlit-folium geopandas
```

3. Ensure you have the required files:
- `liquor_dashboard.py`: Main dashboard code
- `iowa-counties.geojson`: County boundary data for mapping

4. Run the dashboard:
```bash
streamlit run liquor_dashboard.py
```

## Data Source

The dashboard uses Iowa liquor sales data hosted on Google Drive, automatically loaded when you run the application.

## Usage

1. Select different tabs to explore various aspects of the data:
   - Personal Bio: Learn about the project
   - County Topline: Analyze county-level performance
   - Sales Comparisons: Compare categories and stores
   - Forecasting: View sales predictions

2. Use interactive features:
   - County selector
   - Date range picker
   - Category filters
   - Interactive maps and charts

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Plotly Express
- Folium
- Streamlit-Folium
- GeoPandas

## Author

Josh Kozak