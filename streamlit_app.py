# streamlit app

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.oauth2 import service_account
from google.cloud import bigquery
from src.funnel import funnel
import plotly.express as px

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)

client = bigquery.Client(credentials=credentials)

query = """
WITH visitors AS (
SELECT *,
  -- (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'ga_session_number') AS ga_session_id,
  concat(DATE(TIMESTAMP_MICROS(event_timestamp), "America/New_York"),user_pseudo_id, " ", (SELECT value.int_value FROM unnest(event_params) WHERE key = "ga_session_id")) AS session_id
FROM `rusty-creek-coffee.analytics_437086730.events_2024*`
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

# tmp = pd.read_gbq(query, credentials=credentials)
# tmp

# Website title
st.title("Coffee Sales Analysis :coffee:")


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


df = pandas_query(query)

# st.dataframe(df.head())

check = st.button('Update data')
if check:
    df = pandas_query(query)
# st.dataframe(df)
st.success("Downloaded fresh data!")


# Print results.

# st.write("Funnel analysis:")

# if __name__ == '__main__':

#     st.subheader('Download updated sales data')

#     # sel = st.radio("Select option", ['run_query', 'pandas'])


#     to_plot = st.button('Plot a funnel graph')

#     if to_plot:
#         arr = np.random.normal(1, 1, size=100)
#         fig, ax = plt.subplots()
#         ax.hist(arr, bins=20)
#         st.pyplot(fig)

# st.header("This is the header")
# st.markdown("This is the markdown")
# st.subheader("This is the subheader")
# st.caption("This is the caption")
# st.code("x = 2021")
# st.latex(r''' a+a r^1+a r^2+a r^3 ''')


# st.checkbox('Yes')
# st.button('Click Me')
# st.radio('Pick your gender', ['Male', 'Female'])
# st.selectbox('Pick a fruit', ['Apple', 'Banana', 'Orange'])
# st.multiselect('Choose a planet', ['Jupiter', 'Mars', 'Neptune'])
# st.select_slider('Pick a mark', ['Bad', 'Good', 'Excellent'])


st.header("Funnel analysis")

st.markdown("A funnel analysis tracks the number of visitors to a site and follows them through from when they first land on the homepage to when they purchase an item. Places where you see a big drop-off from one level of the funnel to the next indicate potential bottlenecks in user behavior/website design that could be addressed. There is an additional option below to color the plot by region/state.")

# Funnel analysis without regions
df_funnel = funnel(df)
df_funnel.reset_index(inplace=True)

# Funnel analysis with regions
df_funnel_regions = df.groupby('region').apply(funnel)
df_funnel_regions.reset_index(inplace=True)

# Select top-3 regions with most site interaction
# top3 = df_funnel.groupby('region')['count'].sum().nlargest(3)
# top3_names = top3.index.tolist()

# st.dataframe(top3)

# Replace values for better plots
df_funnel['event_new'] = df_funnel['event'].replace({'page_view': 'Page views', 'view_item': 'Item views', 'add_to_cart': 'Cart adds', 'purchase': 'Purchases'})
df_funnel_regions['event_new'] = df_funnel_regions['event'].replace({'page_view': 'Page views', 'view_item': 'Item views', 'add_to_cart': 'Cart adds', 'purchase': 'Purchases'})

# Pick regions of interest
picks = df_funnel_regions['region'].unique()

idx = st.multiselect('Choose states:', picks)

# Create a funnel plot
if len(idx) == 0:
    fig = px.funnel(df_funnel, x='count', y='event_new')
else:
    fig = px.funnel(df_funnel_regions[df_funnel_regions['region'].isin(idx)], x='count', y='event_new', color='region')

fig.update_yaxes(title=None)
st.plotly_chart(fig)

num_days_past = len(df['date'].unique())

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

num_days_future = st.slider('Days out for forecasting', 0, 365)

### Prophet!

from prophet import Prophet

coffee_sales = df.groupby(['date'])[['revenue']].apply('sum')
coffee_sales.reset_index(inplace=True)
coffee_sales.rename(columns={'date': 'ds', 'revenue': 'y'}, inplace=True)

# Fit prophet model
coffee_sales_model = Prophet(interval_width=0.95)
coffee_sales_model.fit(coffee_sales)

# Forecast using the model
coffee_sales_forecast = coffee_sales_model.make_future_dataframe(periods=num_days_future, freq='D')
coffee_sales_forecast = coffee_sales_model.predict(coffee_sales_forecast)

plt.figure(figsize=(18, 6))
ax = coffee_sales_model.plot(coffee_sales_forecast, xlabel = 'Date', ylabel = 'Sales', include_legend=True)
plt.title('Coffee Sales');

# Projected daily sales 12 months from now
sales_next_year = coffee_sales_forecast['yhat'].tolist()[-1]

# Projected total annual sales
sales_sum_year = coffee_sales_forecast['yhat'].sum()

st.pyplot(ax)


if num_days_future == 0:
    st.write(f"Total revenue over the last {num_days_past} days: ${sales_sum_year:,.2f} :moneybag:")
else:
    st.write(f"According to the model, the projected daily sales {num_days_past} days from now will be: ${sales_next_year:,.2f}")
    st.write(f"Total projected revenue over the next {num_days_future} days: ${sales_sum_year:,.2f} :moneybag:")

