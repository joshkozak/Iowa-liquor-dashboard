import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import traceback
from datetime import datetime
import numpy as np

# Set page config
st.set_page_config(
    page_title="Iowa Liquor Sales Dashboard",
    page_icon="🥃",
    layout="wide"
)

# Helper Functions
@st.cache_data
def load_data():
    try:
        # Read the CSV file
        df = pd.read_csv('iowa_sales.csv')
        
        # Rename columns to match our code
        df = df.rename(columns={
            'Date': 'date',
            'Store Name': 'store_name',
            'County': 'county',
            'Category Name': 'category',
            'Sale (Dollars)': 'sale_dollars'
        })
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Ensure sale_dollars is numeric and county is string
        df['sale_dollars'] = pd.to_numeric(df['sale_dollars'], errors='coerce')
        df['county'] = df['county'].astype(str)
        
        # Clean county names
        df['county'] = df['county'].str.strip().str.upper()
        
        # Remove any rows where sale_dollars is null
        df = df.dropna(subset=['sale_dollars'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_top_stores(df, county):
    try:
        county_data = df[df['county'].astype(str) == str(county)]
        store_sales = county_data.groupby('store_name')['sale_dollars'].sum().reset_index()
        return store_sales.nlargest(10, 'sale_dollars')
    except Exception as e:
        st.error(f"Error in get_top_stores: {str(e)}")
        return pd.DataFrame()

def get_county_categories(df):
    try:
        return df.groupby(['county', 'category'])['sale_dollars'].sum().reset_index()
    except Exception as e:
        st.error(f"Error in get_county_categories: {str(e)}")
        return pd.DataFrame()

# Load data once
df = load_data()

# Create tabs
tabs = st.tabs(["Personal Bio", "County Topline", "Sales Comps", "Forecasting"])

# Global sidebar filter
if df is not None:
    # Show filter for County Topline and Sales Comps tabs
    with st.sidebar:
        if not st.session_state.get('current_tab') == "Personal Bio":
            st.header("Filters")
            counties = sorted(df['county'].unique().astype(str).tolist())
            selected_county = st.selectbox(
                "Select County",
                options=counties,
                key="global_county_filter"
            )

# Tab 1: Personal Bio
with tabs[0]:
    st.session_state['current_tab'] = "Personal Bio"
    st.sidebar.empty()
    
    st.header("About Me")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Professional Summary")
        st.write("""
        I am a data analyst with expertise in Python, SQL, and data visualization. 
        My focus is on transforming complex data into actionable insights that drive business decisions.
        """)
        
        st.subheader("Skills")
        st.write("""
        - Python
        - SQL
        - Data Visualization
        - Statistical Analysis
        - Machine Learning
        """)
        
        st.subheader("Experience")
        st.write("""
        - **Data Analyst** | XYZ Company (2020-Present)
          - Led data analysis projects resulting in 25% efficiency improvement
          - Developed automated reporting systems using Python
        
        - **Junior Analyst** | ABC Corp (2018-2020)
          - Conducted market research and competitor analysis
          - Created monthly performance dashboards
        """)
        
    with col2:
        st.subheader("Education")
        st.write("""
        - **Master's in Data Analytics**
          - University of Data Science
          - 2018
        
        - **Bachelor's in Statistics**
          - Analytics University
          - 2016
        """)
        
        st.subheader("Certifications")
        st.write("""
        - Google Data Analytics
        - IBM Data Science Professional
        - Python for Data Science
        """)
        
        st.subheader("Contact")
        st.write("""
        - 📧 email@example.com
        - 🔗 linkedin.com/in/username
        - 🌐 portfolio-website.com
        """)

        # Tab 2: County Topline
with tabs[1]:
    st.session_state['current_tab'] = "County Topline"
    try:
        if df is not None:
            # Filter data for selected county
            county_data = df[df['county'].astype(str) == str(st.session_state.global_county_filter)]
            
            # 1. Monthly Sales Trend
            st.subheader(f"Monthly Sales Trend - {st.session_state.global_county_filter} County")
            monthly_sales = county_data.groupby(pd.Grouper(key='date', freq='M'))['sale_dollars'].sum().reset_index()
            fig_line = px.line(monthly_sales, x='date', y='sale_dollars',
                             labels={'date': 'Month', 'sale_dollars': 'Sales ($)'},
                             title=f'Monthly Sales Trend - {st.session_state.global_county_filter} County')
            fig_line.update_traces(line_color='#00ff00')
            fig_line.update_layout(
                xaxis_title="Month",
                yaxis_title="Sales ($)",
                plot_bgcolor='#1f1f1f',
                paper_bgcolor='#1f1f1f',
                font_color='white',
                xaxis=dict(gridcolor='#303030'),
                yaxis=dict(gridcolor='#303030')
            )
            st.plotly_chart(fig_line, use_container_width=True)

            # 2. Top 10 Stores Table
            st.subheader(f"Top 10 Stores by Sales - {st.session_state.global_county_filter} County")
            top_stores = get_top_stores(df, st.session_state.global_county_filter)
            top_stores['sale_dollars'] = top_stores['sale_dollars'].apply(lambda x: f"${x:,.2f}")
            st.table(top_stores)

            # 3. Top 8 Categories Bar Chart
            st.subheader(f"Top 8 Categories by Sales - {st.session_state.global_county_filter} County")
            category_sales = county_data.groupby('category')['sale_dollars'].sum().reset_index()
            top_categories = category_sales.nlargest(8, 'sale_dollars').sort_values('sale_dollars')
            
            fig_bar = go.Figure(go.Bar(
                x=top_categories['sale_dollars'],
                y=top_categories['category'],
                orientation='h',
                text=top_categories['sale_dollars'].apply(lambda x: f"${x:,.2f}"),
                textposition='auto',
                marker_color='#00ff00'
            ))
            
            fig_bar.update_layout(
                title=f'Top 8 Categories by Sales - {st.session_state.global_county_filter} County',
                xaxis_title="Sales ($)",
                yaxis_title="Category",
                plot_bgcolor='#1f1f1f',
                paper_bgcolor='#1f1f1f',
                font_color='white',
                height=400,
                xaxis=dict(gridcolor='#303030'),
                yaxis=dict(gridcolor='#303030')
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # 4. County Map
            st.subheader("County Map - Highest Selling Category by Sales Dollars")
            try:
                iowa_counties = gpd.read_file(r"C:\Users\joshk\iowa-counties.geojson")
                iowa_counties = iowa_counties[iowa_counties['StateAbbr'] == 'IA']
                
                county_categories = df.groupby(['county', 'category'])['sale_dollars'].sum().reset_index()
                top_categories_by_county = (county_categories.sort_values('sale_dollars', ascending=False)
                                         .groupby('county').first().reset_index())
                
                unique_categories = sorted(top_categories_by_county['category'].unique())
                colors = px.colors.qualitative.Set3[:len(unique_categories)]
                category_colors = dict(zip(unique_categories, colors))
                
                m = folium.Map(location=[42.0, -93.5], zoom_start=7)
                
                for idx, row in iowa_counties.iterrows():
                    county_name = row['CountyName']
                    if isinstance(county_name, str):
                        county_name = county_name.strip().upper()
                    
                    if county_name != 'UNKNOWN':
                        county_data = top_categories_by_county[top_categories_by_county['county'] == county_name]
                        
                        if not county_data.empty:
                            category = county_data.iloc[0]['category']
                            sales = county_data.iloc[0]['sale_dollars']
                            color = category_colors[category]
                            
                            geojson = folium.GeoJson(
                                row.geometry.__geo_interface__,
                                name=f"county_{idx}",
                                style_function=lambda x, color=color: {
                                    'fillColor': color,
                                    'color': 'black',
                                    'weight': 1,
                                    'fillOpacity': 0.7
                                },
                                tooltip=folium.Tooltip(
                                    f"<b>{county_name}</b><br>"
                                    f"Top Category: {category}<br>"
                                    f"Sales: ${sales:,.2f}"
                                )
                            )
                            geojson.add_to(m)
                        else:
                            geojson = folium.GeoJson(
                                row.geometry.__geo_interface__,
                                name=f"county_{idx}",
                                style_function=lambda x: {
                                    'fillColor': '#808080',
                                    'color': 'black',
                                    'weight': 1,
                                    'fillOpacity': 0.3
                                },
                                tooltip=folium.Tooltip(f"<b>{county_name}</b><br>No sales data")
                            )
                            geojson.add_to(m)

                legend_html = """
                <div style="position: fixed; bottom: 50px; right: 50px; z-index: 1000; background-color: white; 
                            padding: 10px; border: 2px solid grey; border-radius: 5px">
                <p><b>Top Selling Category</b></p>
                """
                for category in unique_categories:
                    color = category_colors[category]
                    legend_html += f'<p><span style="background-color: {color}; padding: 0 10px">&nbsp;</span> {category}</p>'
                legend_html += '<p><span style="background-color: #808080; padding: 0 10px">&nbsp;</span> No Data</p>'
                legend_html += "</div>"
                
                m.get_root().html.add_child(folium.Element(legend_html))
                
                st_folium(m, key="county_map", width=700, height=500)

            except Exception as e:
                st.error(f"Error creating map: {str(e)}")
                st.error(f"Full error: {traceback.format_exc()}")

    except Exception as e:
        st.error(f"Error in County Topline tab: {str(e)}")

# Tab 3: Sales Comps
with tabs[2]:
    st.session_state['current_tab'] = "Sales Comps"
    try:
        if df is not None:
            # Filter data for selected county
            county_data = df[df['county'].astype(str) == str(st.session_state.global_county_filter)]
            
            # Calculate monthly sales for the last two years
            monthly_sales = county_data.groupby([
                pd.Grouper(key='date', freq='M')
            ])['sale_dollars'].sum().reset_index()
            
            # Create year and month columns
            monthly_sales['year'] = monthly_sales['date'].dt.year
            monthly_sales['month'] = monthly_sales['date'].dt.month
            monthly_sales['month_name'] = monthly_sales['date'].dt.strftime('%B')
            
            # Get the most recent year in the data
            max_year = monthly_sales['year'].max()
            prev_year = max_year - 1
            
            # Create complete datasets for both years
            months = ['January', 'February', 'March', 'April', 'May', 'June',
                     'July', 'August', 'September', 'October', 'November', 'December']
            
            prev_year_data = pd.DataFrame({'month_name': months})
            curr_year_data = pd.DataFrame({'month_name': months})
            
            prev_year_sales = monthly_sales[monthly_sales['year'] == prev_year]
            curr_year_sales = monthly_sales[monthly_sales['year'] == max_year]
            
            prev_year_data = prev_year_data.merge(
                prev_year_sales[['month_name', 'sale_dollars']], 
                on='month_name', 
                how='left'
            ).fillna(0)
            
            curr_year_data = curr_year_data.merge(
                curr_year_sales[['month_name', 'sale_dollars']], 
                on='month_name', 
                how='left'
            ).fillna(0)
            
            # Create the comparison bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name=str(prev_year),
                x=prev_year_data['month_name'],
                y=prev_year_data['sale_dollars'],
                text=prev_year_data['sale_dollars'].apply(lambda x: f"${x:,.0f}"),
                textposition='auto',
                marker_color='#004d99'
            ))
            
            fig.add_trace(go.Bar(
                name=str(max_year),
                x=curr_year_data['month_name'],
                y=curr_year_data['sale_dollars'],
                text=curr_year_data['sale_dollars'].apply(lambda x: f"${x:,.0f}"),
                textposition='auto',
                marker_color='#00ff00'
            ))
            
            fig.update_layout(
                title=f'Monthly Sales Comparison - {st.session_state.global_county_filter} County',
                xaxis_title="Month",
                yaxis_title="Sales ($)",
                plot_bgcolor='#1f1f1f',
                paper_bgcolor='#1f1f1f',
                font_color='white',
                height=600,
                barmode='group',
                xaxis=dict(
                    gridcolor='#303030',
                    categoryorder='array',
                    categoryarray=months
                ),
                yaxis=dict(gridcolor='#303030'),
                showlegend=True,
                legend_title_text="Year",
                bargap=0.2,
                bargroupgap=0.1
            )
            
            for month in months:
                prev_sale = prev_year_data[prev_year_data['month_name'] == month]['sale_dollars'].values[0]
                curr_sale = curr_year_data[curr_year_data['month_name'] == month]['sale_dollars'].values[0]
                
                if prev_sale > 0:
                    pct_change = ((curr_sale - prev_sale) / prev_sale) * 100
                    fig.add_annotation(
                        x=month,
                        y=max(curr_sale, prev_sale),
                        text=f"{pct_change:+.1f}%",
                        showarrow=False,
                        yshift=10,
                        font=dict(color='white')
                    )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary table
            st.subheader("Monthly Sales Summary")
            summary_data = pd.DataFrame({
                'Month': months,
                f'{prev_year} Sales': prev_year_data['sale_dollars'].apply(lambda x: f"${x:,.2f}"),
                f'{max_year} Sales': curr_year_data['sale_dollars'].apply(lambda x: f"${x:,.2f}"),
                'YoY Change %': [
                    f"{((c - p) / p * 100):+.1f}%" if p > 0 else "N/A"
                    for c, p in zip(curr_year_data['sale_dollars'], prev_year_data['sale_dollars'])
                ]
            })
            st.table(summary_data)
            
    except Exception as e:
        st.error(f"Error in Sales Comps tab: {str(e)}")
        st.error(f"Full error: {traceback.format_exc()}")


# Tab 4: Forecasting
with tabs[3]:
    st.session_state['current_tab'] = "Forecasting"
    try:
        if df is not None:
            # Sidebar filters
            with st.sidebar:
                st.header("Forecast Filters")
                categories = sorted(df['category'].unique().tolist())
                selected_category = st.selectbox(
                    "Select Category",
                    options=categories,
                    key="forecast_category"
                )
            
            # Filter data for selected category
            category_data = df[df['category'] == selected_category].copy()
            
            # Add month and year columns
            category_data['month'] = category_data['date'].dt.month
            category_data['year'] = category_data['date'].dt.year
            
            # Calculate seasonal indices
            def calculate_seasonal_indices(data):
                # Ensure we have data for all months
                all_months = pd.DataFrame({'month': range(1, 13)})
                monthly_data = data.groupby('month')['Bottles Sold'].mean().reset_index()
                monthly_data = all_months.merge(monthly_data, on='month', how='left')
                monthly_data['Bottles Sold'] = monthly_data['Bottles Sold'].fillna(method='ffill').fillna(method='bfill')
                
                # Calculate seasonal indices
                overall_avg = monthly_data['Bottles Sold'].mean()
                if overall_avg == 0:
                    return pd.Series(1, index=range(1, 13))  # Return flat seasonality if no sales
                
                seasonal_indices = monthly_data['Bottles Sold'] / overall_avg
                return pd.Series(seasonal_indices.values, index=range(1, 13))
            
            # Calculate seasonal indices for each item
            seasonal_patterns = {}
            for item in category_data['Item Description'].unique():
                item_data = category_data[category_data['Item Description'] == item]
                seasonal_patterns[item] = calculate_seasonal_indices(item_data)
            
            # Calculate base velocity for each item
            velocity_calc = (category_data.groupby('Item Description')
                           .agg({
                               'store_name': 'nunique',
                               'Bottles Sold': 'sum'
                           })
                           .reset_index())
            
            # Calculate time period in months
            time_range = max((df['date'].max() - df['date'].min()).days / 30.44, 1)
            
            # Calculate monthly base velocity
            velocity_calc['Base_Monthly_Velocity'] = velocity_calc['Bottles Sold'] / time_range
            
            # Generate future months
            last_date = df['date'].max()
            future_months = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=12,
                freq='M'
            )
            
            # Create forecast with seasonality
            forecasts = []
            for _, row in velocity_calc.iterrows():
                item = row['Item Description']
                base_velocity = row['Base_Monthly_Velocity']
                seasonal_idx = seasonal_patterns[item]
                
                item_forecast = pd.DataFrame({
                    'Item Description': [item] * 12,
                    'Month': future_months,
                    'Forecasted_Bottles': [
                        round(base_velocity * seasonal_idx[m])
                        for m in future_months.month
                    ]
                })
                forecasts.append(item_forecast)
            
            forecast_df = pd.concat(forecasts, ignore_index=True)
            
            # Pivot the data for display
            forecast_pivot = forecast_df.pivot(
                index='Item Description',
                columns='Month',
                values='Forecasted_Bottles'
            )
            
            # Format column names as month and year
            forecast_pivot.columns = forecast_pivot.columns.strftime('%B %Y')
            
            # Add historical data columns
            historical_totals = category_data.groupby('Item Description')['Bottles Sold'].sum()
            historical_monthly_avg = category_data.groupby('Item Description')['Bottles Sold'].mean()
            
            forecast_pivot.insert(0, 'Historical Total', historical_totals)
            forecast_pivot.insert(1, 'Monthly Average', historical_monthly_avg)
            
            # Calculate and add seasonality strength
            def calculate_seasonality_strength(item_data):
                if len(item_data) < 12:
                    return "Insufficient Data"
                monthly_std = item_data.groupby('month')['Bottles Sold'].mean().std()
                overall_std = item_data['Bottles Sold'].std()
                if overall_std == 0:
                    return "No Variation"
                return f"{(monthly_std/overall_std):,.2f}"
            
            seasonality_strength = {
                item: calculate_seasonality_strength(category_data[category_data['Item Description'] == item])
                for item in category_data['Item Description'].unique()
            }
            
            forecast_pivot.insert(2, 'Seasonality Strength', pd.Series(seasonality_strength))
            
            # Format the table
            st.header(f"Monthly Forecast - {selected_category}")
            st.markdown("""
            This forecast incorporates seasonal patterns based on historical monthly sales variations.
            The Seasonality Strength indicates how much an item's sales vary by month (higher numbers indicate stronger seasonal patterns).
            """)
            
            # Style the dataframe
            def highlight_rows(row):
                return ['background-color: #1f1f1f; color: white'] * len(row)
            
            styled_forecast = forecast_pivot.style\
                .apply(highlight_rows, axis=1)\
                .format({
                    'Historical Total': '{:,.0f}',
                    'Monthly Average': '{:,.1f}',
                    **{col: '{:,.0f}' for col in forecast_pivot.columns[3:]}
                })
            
            # Display the forecast table
            st.dataframe(
                styled_forecast,
                use_container_width=True,
                height=600
            )
            
            # Add visualization of seasonal patterns
            st.subheader("Seasonal Patterns")
            
            # Create seasonal pattern chart for top items
            top_items = velocity_calc.nlargest(5, 'Bottles Sold')['Item Description'].tolist()
            
            fig = go.Figure()
            
            for item in top_items:
                seasonal_idx = seasonal_patterns[item]
                fig.add_trace(go.Scatter(
                    x=list(range(1, 13)),
                    y=seasonal_idx.values,
                    name=item,
                    mode='lines+markers'
                ))
            
            fig.update_layout(
                title='Seasonal Patterns for Top 5 Items',
                xaxis_title='Month',
                yaxis_title='Seasonal Index',
                plot_bgcolor='#1f1f1f',
                paper_bgcolor='#1f1f1f',
                font_color='white',
                xaxis=dict(
                    gridcolor='#303030',
                    tickmode='array',
                    ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                    tickvals=list(range(1, 13))
                ),
                yaxis=dict(gridcolor='#303030'),
                showlegend=True,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_stores = category_data['store_name'].nunique()
                st.metric("Total Stores Selling Category", f"{total_stores:,}")
            
            with col2:
                total_bottles = category_data['Bottles Sold'].sum()
                st.metric("Total Historical Bottles Sold", f"{total_bottles:,}")
            
            with col3:
                avg_monthly = total_bottles / time_range
                st.metric("Average Monthly Bottles", f"{avg_monthly:,.0f}")
            
            # Add methodology explanation
            with st.expander("Forecast Methodology"):
                st.markdown("""
                ### Forecast Calculation Method
                
                1. **Seasonal Pattern Analysis**:
                   - Calculate monthly sales patterns for each item
                   - Determine seasonal indices showing typical monthly variations
                   - Measure seasonality strength to identify items with consistent patterns
                
                2. **Base Velocity Calculation**:
                   - Calculate average monthly sales per item
                   - Account for number of stores selling each item
                   - Establish baseline demand levels
                
                3. **Forecast Generation**:
                   - Apply seasonal patterns to base velocity
                   - Project forward for next 12 months
                   - Adjust predictions based on historical monthly variations
                
                4. **Seasonality Strength**:
                   - Higher values indicate stronger seasonal patterns
                   - Values near 0 suggest consistent month-to-month sales
                   - "Insufficient Data" shown for items with limited history
                """)
            
    except Exception as e:
        st.error(f"Error in Forecasting tab: {str(e)}")
        st.error(f"Full error: {traceback.format_exc()}")