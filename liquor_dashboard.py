import streamlit as st
st.set_page_config(
    page_title="Iowa Liquor Sales Dashboard",
    page_icon="🥃",
    layout="wide"
)

import pandas as pd
import numpy as np
try:
    import plotly.express as px
except ImportError:
    import pip
    pip.main(['install', 'plotly'])
    import plotly.express as px
import folium
from streamlit_folium import st_folium
import geopandas as gpd

def clean_county_name(name):
    """Clean county names to make them consistent"""
    name = str(name).strip()
    # Handle specific cases
    name_map = {
        "O'BRIEN": "OBRIEN",
        "O BRIEN": "OBRIEN",
        "OBRIEN": "OBRIEN",
        "BUENA VISTA": "BUENAVISTA",
        "CERRO GORDO": "CERROGORDO",
        "DES MOINES": "DESMOINES",
        "POTTAWATTAMIE": "POTTAWATAMIE",
        "POTTAWATAMIE": "POTTAWATAMIE",
        "VAN BUREN": "VANBUREN"
    }
    name = name.upper()
    return name_map.get(name, name)

def load_data():
    try:
        file_id = "1Chmj-7ZmQ6usxQeMO-g32K72QZ4bS8FI"
        url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
        
        df = pd.read_csv(url)
        df['Date'] = pd.to_datetime(df['Date'])
        df['County'] = df['County'].apply(clean_county_name)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


df = load_data()

if df is not None:
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Personal Bio", "County Topline", "Sales Comparisons", "Forecasting"])

    # Personal Bio tab
    with tab1:
        st.title("About Me")
        st.write("""
        Hi, I'm Josh Kozak! I'm a data enthusiast with a passion for turning raw data into meaningful insights.
        This dashboard explores Iowa liquor sales data, showcasing various analyses and visualizations.
        
        Key Features:
        - County-level sales analysis
        - Store performance metrics
        - Category popularity trends
        - Sales forecasting
        """)

    # County Topline tab
    with tab2:
        st.title("County Analysis")
        
        # Sidebar for county selection
        counties = sorted(df['County'].unique().astype(str).tolist())
        selected_county = st.selectbox("Select County", counties)
        
        # Filter data for selected county
        county_data = df[df['County'] == selected_county]
        
        # Monthly sales trend
        monthly_sales = county_data.groupby(county_data['Date'].dt.to_period('M')).agg({
            'Sale (Dollars)': 'sum'
        }).reset_index()
        monthly_sales['Date'] = monthly_sales['Date'].astype(str)
        
        # Create line chart
        fig_trend = px.line(monthly_sales, x='Date', y='Sale (Dollars)',
                           title=f'Monthly Sales Trend - {selected_county} County')
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Create two columns for stores and categories
        col1, col2 = st.columns(2)
        
        with col1:
            # Top stores table
            top_stores = county_data.groupby(['Store Name', 'City']).agg({
                'Sale (Dollars)': 'sum'
            }).reset_index().sort_values('Sale (Dollars)', ascending=False).head(10)
            
            st.subheader(f"Top 10 Stores in {selected_county} County")
            st.dataframe(top_stores)
        
        with col2:
            # Top categories bar chart
            top_categories = county_data.groupby('Category Name').agg({
                'Sale (Dollars)': 'sum'
            }).reset_index().sort_values('Sale (Dollars)', ascending=False).head(8)
            
            st.subheader(f"Top 8 Categories by Sales - {selected_county} County")
            fig_categories = px.bar(top_categories, x='Category Name', y='Sale (Dollars)')
            st.plotly_chart(fig_categories, use_container_width=True)
        
        # Map section at the bottom
        st.subheader("Top Selling Categories by County")
        try:
            # Load GeoJSON file with full path

            gdf = gpd.read_file('C:/Users/joshk/Iowa-Liquor-Dashboard/iowa-counties.geojson')

            gdf = gpd.read_file('C:/Users/joshk/IA-Liquor/iowa-counties.geojson')
            # Clean county names in GeoJSON
            gdf['CountyName'] = gdf['CountyName'].apply(clean_county_name)
            
            # Calculate highest selling category per county
            county_categories = df.groupby(['County', 'Category Name'])['Sale (Dollars)'].sum().reset_index()
            top_categories_per_county = county_categories.loc[county_categories.groupby('County')['Sale (Dollars)'].idxmax()]
            
            # Create map
            m = folium.Map(location=[42.0046, -93.2140], zoom_start=7)
            
            # Create a color map for categories
            unique_categories = top_categories_per_county['Category Name'].unique()
            color_map = {cat: px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)] 
                        for i, cat in enumerate(unique_categories)}
            
            # Add counties to map
            for idx, row in gdf.iterrows():
                county_name = row['CountyName']
                category_info = top_categories_per_county[top_categories_per_county['County'] == county_name]
                
                if not category_info.empty:
                    category = category_info.iloc[0]['Category Name']
                    sales = category_info.iloc[0]['Sale (Dollars)']
                    color = color_map[category]
                else:
                    category = "No Data"
                    sales = 0
                    color = '#808080'
                
                folium.GeoJson(
                    row.geometry,
                    style_function=lambda x, color=color: {
                        'fillColor': color,
                        'fillOpacity': 0.7,
                        'color': 'black',
                        'weight': 1
                    },
                    tooltip=f"{county_name}<br>Top Category: {category}<br>Sales: ${sales:,.2f}"
                ).add_to(m)
            
            # Add a legend
            legend_html = '''
                <div style="position: fixed; 
                            bottom: 50px; right: 50px; 
                            border:2px solid grey; z-index:9999; 
                            background-color:white;
                            padding:10px;
                            border-radius:6px;">
                <p><b>Top Selling Categories</b></p>
            '''
            for category, color in color_map.items():
                legend_html += f'<p><i style="background: {color}; width:20px; height:20px; display:inline-block"></i> {category}</p>'
            legend_html += '</div>'
            m.get_root().html.add_child(folium.Element(legend_html))
            
            # Display map
            st_folium(m, height=600)
            
        except Exception as e:
            st.error(f"Error creating map: {e}")
            st.write("Full error:", str(e))

    # Sales Comparisons tab
    with tab3:
        st.title("Sales Comparisons")
        
        # Date range selector
        date_range = st.date_input(
            "Select Date Range",
            value=(df['Date'].min(), df['Date'].max()),
            min_value=df['Date'].min().to_pydatetime(),
            max_value=df['Date'].max().to_pydatetime()
        )
        
        # Filter data by date range
        mask = (df['Date'] >= pd.Timestamp(date_range[0])) & (df['Date'] <= pd.Timestamp(date_range[1]))
        filtered_df = df.loc[mask]
        
        # Create two columns for metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # Category comparison
            st.subheader("Category Comparison")
            top_categories = filtered_df.groupby('Category Name')['Sale (Dollars)'].sum().sort_values(ascending=False)
            fig_categories = px.bar(
                x=top_categories.index,
                y=top_categories.values,
                title="Sales by Category",
                labels={'x': 'Category', 'y': 'Total Sales ($)'}
            )
            fig_categories.update_layout(showlegend=False)
            st.plotly_chart(fig_categories, use_container_width=True)
        
        with col2:
            # Store comparison
            st.subheader("Store Comparison")
            top_stores = filtered_df.groupby('Store Name')['Sale (Dollars)'].sum().sort_values(ascending=False).head(20)
            fig_stores = px.bar(
                x=top_stores.index,
                y=top_stores.values,
                title="Top 20 Stores by Sales",
                labels={'x': 'Store', 'y': 'Total Sales ($)'}
            )
            fig_stores.update_layout(showlegend=False)
            st.plotly_chart(fig_stores, use_container_width=True)
        
        # Monthly trend
        st.subheader("Monthly Sales Trend")
        monthly_trend = filtered_df.groupby(filtered_df['Date'].dt.to_period('M')).agg({
            'Sale (Dollars)': 'sum',
            'Bottles Sold': 'sum'
        }).reset_index()
        monthly_trend['Date'] = monthly_trend['Date'].astype(str)
        
        fig_trend = px.line(
            monthly_trend,
            x='Date',
            y=['Sale (Dollars)', 'Bottles Sold'],
            title="Monthly Sales and Volume Trend",
            labels={'value': 'Amount', 'variable': 'Metric'}
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # Forecasting tab
    with tab4:
        st.title("Sales Forecasting")
        
        # Category selector
        categories = sorted(df['Category Name'].unique())
        selected_category = st.selectbox("Select Category for Forecast", categories, key='forecast_category')
        
        # Filter data for selected category
        category_data = df[df['Category Name'] == selected_category].copy()
        
        # Add month and year columns
        category_data['Year'] = category_data['Date'].dt.year
        category_data['Month'] = category_data['Date'].dt.month
        
        # Calculate seasonal indices
        monthly_sales = category_data.groupby(['Year', 'Month'])['Bottles Sold'].sum().reset_index()
        
        # Ensure all months are represented
        all_months = pd.DataFrame({'Month': range(1, 13)})
        seasonal_idx = monthly_sales.groupby('Month')['Bottles Sold'].mean()
        seasonal_idx = pd.merge(all_months, seasonal_idx.reset_index(), on='Month', how='left')
        seasonal_idx['Bottles Sold'] = seasonal_idx['Bottles Sold'].fillna(seasonal_idx['Bottles Sold'].mean())
        seasonal_idx.set_index('Month', inplace=True)
        seasonal_idx = seasonal_idx['Bottles Sold'] / seasonal_idx['Bottles Sold'].mean()
        
        # Calculate base velocity for each item
        item_data = category_data.groupby(['Item Number', 'Item Description']).agg({
            'Bottles Sold': 'sum',
            'Date': ['min', 'max']
        }).reset_index()
        
        item_data.columns = ['Item Number', 'Item Description', 'Total Bottles', 'Start Date', 'End Date']
        item_data['Time Range (Months)'] = ((item_data['End Date'] - item_data['Start Date']).dt.days / 30.44).round(1)
        item_data['Time Range (Months)'] = item_data['Time Range (Months)'].clip(lower=1)  # Minimum 1 month
        item_data['Base Monthly Velocity'] = item_data['Total Bottles'] / item_data['Time Range (Months)']
        
        # Generate future months
        future_months = pd.date_range(
            start=df['Date'].max(),
            periods=13,
            freq='ME'
        )[1:]  # Skip current month
        
        # Create forecast DataFrame with unique item identifiers
        forecasts = []
        for _, item in item_data.iterrows():
            for date in future_months:
                month = date.month
                forecast = item['Base Monthly Velocity'] * seasonal_idx[month]
                forecasts.append({
                    'Item Key': f"{item['Item Number']} - {item['Item Description']}", # Create unique key
                    'Item Description': item['Item Description'],
                    'Month': date.strftime('%Y-%m'),
                    'Forecasted Bottles': round(forecast)
                })
        
        forecast_df = pd.DataFrame(forecasts)
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Stores",
                f"{len(category_data['Store Number'].unique()):,}",
                help="Number of stores selling this category"
            )
        
        with col2:
            st.metric(
                "Total Bottles Sold",
                f"{category_data['Bottles Sold'].sum():,.0f}",
                help="Total historical bottles sold"
            )
        
        with col3:
            st.metric(
                "Avg Monthly Sales",
                f"{category_data['Bottles Sold'].mean():,.0f}",
                help="Average monthly bottles sold"
            )
        
        # Show seasonality pattern
        st.subheader("Seasonal Pattern")
        fig_seasonal = px.line(
            x=seasonal_idx.index,
            y=seasonal_idx.values,
            labels={'x': 'Month', 'y': 'Seasonal Index'},
            title="Seasonal Sales Pattern"
        )
        fig_seasonal.add_hline(y=1.0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_seasonal, use_container_width=True)
        
        # Show forecasts with unique index
        st.subheader("Forecasted Bottles by Item")
        
        try:
            # Sort and get top items by total forecast
            forecast_summary = forecast_df.groupby('Item Key')['Forecasted Bottles'].sum().sort_values(ascending=False)
            top_items = forecast_summary.head(20).index  # Get top 20 items
            
            # Filter forecast_df for top items only
            top_forecast_df = forecast_df[forecast_df['Item Key'].isin(top_items)]
            
            # Create pivot table with top items only
            pivot_df = top_forecast_df.pivot(
                index='Item Key',
                columns='Month',
                values='Forecasted Bottles'
            )
            
            # Clean up the display names but keep unique index
            display_names = pivot_df.index.map(lambda x: x.split(' - ')[1])
            
            # Display the dataframe
            st.write("Top 20 Items by Forecasted Volume:")
            st.dataframe(
                pivot_df.set_index(display_names),
                height=400
            )
            
        except Exception as e:
            st.error(f"Error creating forecast table: {e}")
            st.write("Detailed forecast data:", forecast_df.head())

        # Explanation
        with st.expander("How is the forecast calculated?"):
            st.write("""
            The forecast is calculated using the following method:
            1. **Base Velocity**: Calculate the average monthly sales for each item based on historical data
            2. **Seasonality**: Determine monthly seasonal patterns across all items in the category
            3. **Forecast**: Adjust base velocity by seasonal factors for future months
            
            The seasonal index above 1.0 indicates months with typically higher sales, while below 1.0 indicates slower months.
            
            Note: This is a simplified forecast and actual results may vary based on many factors including:
            - Economic conditions
            - Competition
            - Marketing activities
            - Weather
            - Special events

            """)

            
            

