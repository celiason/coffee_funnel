# streamlit app

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.oauth2 import service_account
from google.cloud import bigquery
from src.funnel import funnel
import plotly.express as px
from prophet import Prophet

# Website title
st.title("Coffee Sales Analysis :coffee:")

# Create API client (secrets on streamlit)
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)

client = bigquery.Client(credentials=credentials)

# Get secrets from streamlit
mysite = st.secrets["gbq_site_name"]

# SQL query for pulling event data from Google Bigquery
query = f"""
WITH visitors AS (
SELECT *,
  -- (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'ga_session_number') AS ga_session_id,
  concat(DATE(TIMESTAMP_MICROS(event_timestamp), "America/New_York"),user_pseudo_id, " ", (SELECT value.int_value FROM unnest(event_params) WHERE key = "ga_session_id")) AS session_id
FROM {mysite}
)
SELECT
  PARSE_DATE('%Y%m%d', event_date) AS date,
  user_pseudo_id,
  session_id,
  event_timestamp,
  event_name,
  LEAD(event_name,1,'end_session') OVER(PARTITION BY session_id ORDER BY event_timestamp) AS event_next,
  geo.region,
  ecommerce.total_item_quantity AS num_items,
  ecommerce.purchase_revenue AS revenue,
  (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'page_title') AS page_title,
  (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'logged_in') AS logged_in
FROM visitors
WHERE event_name IN ('remove_from_cart','page_view','view_item','view_search_results','add_to_cart','purchase')
ORDER BY user_pseudo_id, session_id, event_timestamp
"""

# Perform query
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_data(ttl=3600) # data lives (time to live, ttl) for 3600s = 1h
def pandas_query(query):
    df = pd.read_gbq(query, credentials=credentials)
    df['user_pseudo_id'] = df['user_pseudo_id'].astype(str)
    df['transition'] = df['event_name'] + ' -> ' + df['event_next']
    df['date'] = pd.to_datetime(df['date'])
    # remove NAs
    df = df[df['user_pseudo_id'] != 'None']
    return df

# Load the data
df = pandas_query(query)

# Provide option for user to check for new data
check = st.button('Update data')
if check:
    df = pandas_query(query)
st.success("Downloaded fresh data!")

# Do a funnel analysis
st.header("Funnel analysis")

st.markdown("A funnel analysis tracks the number of visitors to a site and follows them through from when they first land on the homepage to when they purchase an item. Places where you see a big drop-off from one level of the funnel to the next indicate potential bottlenecks in user behavior/website design that could be addressed. There is an additional option below to color the plot by region/state.")

# Funnel analysis without regions
df_funnel = funnel(df)
df_funnel.reset_index(inplace=True)

# Funnel analysis with regions
df_funnel_regions = df.groupby('region').apply(funnel)
df_funnel_regions.reset_index(inplace=True)

# Replace values for better plots
df_funnel['event_new'] = df_funnel['event'].replace({'page_view': 'Page views', 'view_item': 'Item views', 'add_to_cart': 'Cart adds', 'purchase': 'Purchases'})
df_funnel_regions['event_new'] = df_funnel_regions['event'].replace({'page_view': 'Page views', 'view_item': 'Item views', 'add_to_cart': 'Cart adds', 'purchase': 'Purchases'})

# Pick regions of interest
picks = df_funnel_regions['region'].unique()
picks = picks[picks!='']

# TODO: make the multiselect show number of users by state
# df_funnel_regions[df_funnel_regions['event']=='page_view']['count']
# df_funnel_regions[df_funnel_regions['event']=='page_view']['region']

# Here the user can pick states
selected_regions = st.multiselect('Choose states:', picks)

# Create a funnel plot
if len(selected_regions) == 0:
    fig = px.funnel(df_funnel, x='count', y='event_new')
else:
    fig = px.funnel(df_funnel_regions[df_funnel_regions['region'].isin(selected_regions)], x='count', y='event_new', color='region')
fig.update_yaxes(title=None)

# Display the plot
st.plotly_chart(fig)


# Setup sales data
coffee_sales = df.groupby(['date'])[['revenue']].apply('sum')
coffee_sales.reset_index(inplace=True)

# Backfill with old GA4 data
# old_data = pd.read_csv("data/data-export.csv")
# dates = pd.date_range(start='2023-11-06', end='2024-11-06', freq='W')
# old_data['date'] = dates[old_data['Nth week'].values-1]
# old_data['revenue'] = old_data['Total revenue'] / 7
# odf = pd.DataFrame()
# odf['date'] = pd.date_range(start='2023-11-06', end='2024-11-05', freq='D')
# odf = pd.merge_ordered(odf, old_data, fill_method='ffill')

odf = pd.read_csv("data/revenue_by_date.csv")
odf['date'] = pd.to_datetime(odf['date'])

# Do an outer join
coffee_sales = coffee_sales.merge(odf, on='date', how='outer', suffixes=['','_old'])

# Keep new values if not NA
coffee_sales['revenue'] = np.where(coffee_sales['revenue'].isna(), coffee_sales['revenue_old'], coffee_sales['revenue'])
coffee_sales = coffee_sales.dropna(subset='revenue')

# Rename for prophet model
coffee_sales.rename(columns={'date': 'ds', 'revenue': 'y'}, inplace=True)

# Number of days with historic sales data
num_days_past = len(coffee_sales['ds'].unique())


# Show the forecast

st.header("Sale forecasting")

st.write(f"The analysis uses Facebook's `Prophet` package in `python` to run an additive model that incorporates seasonality (daily, weekly). The output of the model is the predict sales over time as well as how much confidence there is in the prediction (shaded regions in the plot below). This model is based on sales data for only {num_days_past} days, so interpret at your own risk. :smile:")

# run_forecast = st.button('Generate sales forecast')

# if __name__ == '__main__':
#     if num > 30:
#         st.markdown("Big number!")
#         fig, ax = plt.subplots()
#         ax.plot(range(0, num))
#         st.pyplot(fig)
#     else:
#         st.markdown("Small number")

# if run_forecast:

freq = st.radio(label='Sales frequency for graphing:', options=['Daily', 'Weekly', 'Monthly'], index=1)

if freq == 'Weekly':
    # Average by week
    df_sales = coffee_sales.groupby(pd.Grouper(key='ds', freq='W')).sum()
    df_sales.reset_index(inplace=True)
elif freq == 'Daily':
    df_sales = coffee_sales
    df_sales.reset_index(inplace=True)
elif freq == 'Monthly':
    # Average by month
    df_sales = coffee_sales.groupby(pd.Grouper(key='ds', freq='M')).sum()
    df_sales.reset_index(inplace=True)

fcast = st.slider('Months out for forecasting:', 0, 36)

# convert mongths to days
num_days_future = int(fcast * 30.4375)

### Prophet!

# For logistic regression

growth_model = 'linear'

if growth_model == 'logistic':
    cap = 1000
    floor = 0
    coffee_sales['cap'] = cap
    coffee_sales['floor'] = floor

# Initiate prophet model
coffee_sales_model = Prophet(interval_width=0.95, growth=growth_model,
                             daily_seasonality=False,
                             weekly_seasonality=False)

# Fit the model
coffee_sales_model.fit(df_sales)

# num_days_future=365

# Forecast using the model
coffee_sales_forecast = coffee_sales_model.make_future_dataframe(periods=num_days_future, freq='D')

if growth_model == 'logistic':
    coffee_sales_forecast['floor'] = floor
    coffee_sales_forecast['cap'] = cap

coffee_sales_forecast = coffee_sales_model.predict(coffee_sales_forecast)
# coffee_sales_model.plot_components(fcst=coffee_sales_forecast)
# plt.savefig('test.png')

plt.figure(figsize=(18, 6))
ax = coffee_sales_model.plot(coffee_sales_forecast, xlabel='Date', ylabel=f'Projected {freq} Revenue', include_legend=True)
plt.plot(df_sales['ds'], df_sales['y'], c='black', label=f'Actual {freq} Revenue', linewidth=0.25)
# plt.title('Coffee Sales')
plt.legend()
st.pyplot(ax)

# A lookup dictionary for converting text to numbers in calcs below
freq_convert = {'Weekly': 7, 'Daily': 1, 'Monthly': 30.4375}

# Projected daily sales 12 months from now
sales_next_year = coffee_sales_forecast['yhat'].tolist()[-1] / freq_convert[freq]

# Total past sales
idx = coffee_sales_forecast['ds'] < coffee_sales['ds'].max()
sales_sum_past = coffee_sales_forecast.loc[idx, 'yhat'].sum()

# Projected total annual sales
idx = coffee_sales_forecast['ds'] > coffee_sales['ds'].max()

# Note we need to scale by the frequency selected above
sales_sum_future = coffee_sales_forecast.loc[idx, 'yhat'].sum() / freq_convert[freq]

# If we only have past data
if num_days_future == 0:
    st.write(f"Total revenue over the last {num_days_past} days: ${sales_sum_past:,.2f} :moneybag:")

# If we are looking into the future
else:
    st.write(f"According to the model, the projected daily sales {num_days_future} days from now will be: ${sales_next_year:,.2f}")
    st.write(f"Total projected revenue over the next {num_days_future} days: ${sales_sum_future:,.2f} :moneybag:")

