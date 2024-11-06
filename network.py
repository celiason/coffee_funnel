#### Cart abandonment analysis

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans

#------------------------------------------------------------------------
# Analysis of conversion rates
#------------------------------------------------------------------------

event_file = '/Users/chad/Downloads/bquxjob_6ff6a5c7_192fd46ff64.csv'

# Load the event data
df_raw = pd.read_csv(event_file)
df_raw['user_pseudo_id'] = df_raw['user_pseudo_id'].astype(str)
df_raw['transition'] = df_raw['event_name'] + ' -> ' + df_raw['event_next']
df_raw['date'] = pd.to_datetime(df_raw['date'])

# remove NAs
df_raw = df_raw[df_raw['user_pseudo_id']!='nan']

# Make a df copy
df2 = df_raw.copy()

len(df2) # 967 rows

# Now for some feature engineering!

# calculate average session time by user
def session_time(df):
    diff_time = max(df['event_timestamp']) - min(df['event_timestamp'])
    return diff_time/1e6  # in seconds
avg_session_dur = df2.groupby(['session_id']).apply(session_time)

# Calculate unique item views
def get_item_views(df):
    return len(set(df[df['event_name']=='view_item']['page_title']))
item_views = df2.groupby('session_id').apply(get_item_views)

# Create summary dataset for plotting
sessions = df2.groupby('session_id').agg({'revenue': sum, 'event_timestamp': 'count', 'region': 'first'})
sessions = pd.concat([sessions, avg_session_dur, item_views], axis=1)
sessions.columns = ['revenue','clicks','region','duration','items']
sessions.dropna(subset = ['revenue','clicks','duration','items'], inplace=True)

# Four-root transform due to lots of zeros
sessions['revenue_4root'] = sessions['revenue'] ** 0.25
sessions['duration_4root'] = sessions['duration'] ** 0.25

# Pairs plot
sns.pairplot(sessions.drop(['revenue','duration'],axis=1))

# Clustering analysis
km = KMeans(n_clusters=5)
km.fit(sessions.drop(['revenue','region','duration'], axis=1))

# Add cluster to plot as color
sessions['cluster'] = km.labels_.astype(str)

sns.pairplot(sessions.drop(['revenue_4root', 'duration_4root'], axis=1), hue='cluster')

# Average revenue by cluster, sorted by descending revenue
sessions.groupby('cluster')[['revenue','clicks','items','duration']].apply('mean').sort_values('revenue', ascending=False)

# total revenue
tot_revenue = df2['revenue'].sum()
print(f"Total revenue: ${tot_revenue}")
# $869.22 total

# daily revenue
num_days = (max(df2['date']) - min(df2['date'])).days
avg_daily_revenue = tot_revenue/num_days
print(f"Average daily revenue: ${round(avg_daily_revenue, 2)}")
# $79.02 per day

# per session average
tmp = df2.groupby('session_id')['revenue'].apply('sum')
print(f"Average revenue per user session: ${round(np.mean(tmp), 2)}")
# $4.91 average per session

# Projected annual revenue
print(f"Projected annual sales: ${365 * tot_revenue/num_days:.2f}")
# $28,842.30 expected annual revenue

# Total units sold
df2['items_sold'] = np.where(df2['event_name'] == 'purchase', df2['num_items'], 0)
units_sold = df2['items_sold'].sum().astype(int)
print(f"Total of {units_sold} items sold over the last {num_days} days")

sales_over_time = df2.groupby(['date','logged_in'])[['revenue', 'items_sold']].apply('sum')

# Line plots showing sales and units sold over time
sns.lineplot(data=sales_over_time, x='date', y='revenue', hue='logged_in')
# sns.lineplot(data=sales_over_time, x='date', y='items_sold', hue='logged_in')

# Fit an ARIMA model and forecast into the future
sales_over_time['day'] = sales_over_time.index.get_level_values('date').dayofweek

# Look at average sales by day of the week (also include standard deviation)
sales_over_time.groupby('day').agg({'revenue': ['mean', 'std']})



### Prophet!

from prophet import Prophet

coffee_sales = df2.groupby(['date'])[['revenue']].apply('sum')
coffee_sales.reset_index(inplace=True)
coffee_sales.rename(columns={'date': 'ds', 'revenue': 'y'}, inplace=True)

# Fit prophet model
coffee_sales_model = Prophet(interval_width=0.95)
coffee_sales_model.fit(coffee_sales)

# Forecast using the model
coffee_sales_forecast = coffee_sales_model.make_future_dataframe(periods=365, freq='D')
coffee_sales_forecast = coffee_sales_model.predict(coffee_sales_forecast)
coffee_sales_forecast.head()
plt.figure(figsize=(18, 6))
coffee_sales_model.plot(coffee_sales_forecast, xlabel = 'Date', ylabel = 'Sales', include_legend=True)
plt.title('Coffee Sales');

# Projected daily sales 12 months from now
sales_next_year = coffee_sales_forecast['yhat'].tolist()[-1]
print(f"Projected daily sales 1 year from now: ${sales_next_year:.2f}")

# Projected total annual sales
sales_sum_year = coffee_sales_forecast['yhat'].sum()
print(f"Projected sales over the next year: ${sales_sum_year:.2f}")
# $613,772.82 - cool! not sure i agree with the trend, but it's working OK.
# NB: get this into a streamlit app!

# Add conversion variable (ie made a purchase)
df2['conversion'] = df2.groupby('session_id')['event_name'].transform(lambda x: np.where(x == 'purchase', 1, 0))
df2['conversion'].value_counts() # 26 conversions (= purchases)



# K-gram encoding
states = ['page_view', 'view_item', 'add_to_cart', 'view_search_results', 'remove_from_cart', 'purchase']

# Create a dictionary mapping text to integers
text_to_int = {text: i for i, text in enumerate(set(states))}


# Make a copy of the df
df_dummy = df2[(df2['event_name'].isin(states))].copy()

df_dummy['conversion'].value_counts() # 22 sales (=1)

df_dummy['event_next'] = np.where(df_dummy['event_next'].isin(states), df_dummy['event_next'], float('nan'))

# Convert to integer
df_dummy['event_name_int'] = df_dummy['event_name'].map(text_to_int)

# Function to 2-gram transitions
def kgrams(df):
    return np.where(df['event_next'].notna(),
                    df['event_name'].map(text_to_int).astype(str) + '' + df['event_next'].map(text_to_int).astype('Int64').astype(str),
                    None)

# Add k-gram feature
df_dummy['2gram'] = kgrams(df_dummy)

# df_dummy['2gram'] = df_dummy['event_name_int'].astype(str) + '' + df_dummy['event_next_int'].astype(str)

df_dummy['2gram'].value_counts()

# Create dummy variables
df_dummy = pd.get_dummies(df_dummy, columns = ['2gram', 'event_name_int'])
df_dummy.columns

xvars = [col for col in df_dummy.columns if '2gram' in col and '0' not in col] + \
        [col for col in df_dummy.columns if 'event_name_int' in col and '0' not in col]

yvar = ['conversion']

# Processed X and y data frames for regression models
X = df_dummy.groupby('session_id')[xvars].sum()
y = df_dummy.groupby('session_id')[yvar].sum()

X['logged_in'] = df_dummy.groupby('session_id')['logged_in'].apply('first')
xvars = xvars + ['logged_in']

# xy = pd.concat([X, y], axis=1)
# sns.scatterplot(data=xy, x='2gram_53', y='conversion')


#------------------------------------------------------------
# Random forest regressor model
#------------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

# Split data into test and train
Xtrain, X_test, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model on the training data
rf_model.fit(Xtrain, ytrain)

# Predict on the test data
y_pred = rf_model.predict(X_test)

# Create the confusion matrix
confusion_matrix(ytest, np.round(y_pred))

# Since ytrain is a DataFrame, we need to flatten it for accuracy_score
accuracy = accuracy_score(ytest, np.round(y_pred))
print(f"Model Accuracy: {accuracy:.2f}")

features = pd.DataFrame(rf_model.feature_importances_)
features.index = xvars
features.sort_values(by=0, ascending=False, inplace=True)
features

text_to_int


## FUNNEL PLOT

"""Create a funnel dataset for plotting
Idea is that we start with a certain number of distinct IDs
then at each subsequent step in the funnel, we remove some

df = data frame with clickstream event data
states = unique states (order is important!)
id_var = unique ID variable to use in the data frame
"""
def funnel(df, states=['page_view','view_item','add_to_cart','purchase'], id_var='user_pseudo_id'):
    idx = set(df[df['event_name'] == states[0]][id_var])
    counts = []
    counts.append(len(idx))
    for state in states[1:]:
        idx_new = set(df[df['event_name'] == state][id_var])
        idx = idx.intersection(idx_new)
        # intersects with last event tier
        size_intersect = len(idx)
        # append
        counts.append(size_intersect)
    df = pd.DataFrame({'event': states, 'count': counts})
    df['last'] = df['count'].shift(1)
    df['dropoff'] = df['count'] - df['last']
    df['prop_last'] = df['count']/df['last']
    df['prop_tot'] = df['count']/df['count'][0]
    return df

df_funnel = df_raw.groupby('region').apply(funnel)
df_funnel.reset_index(inplace=True)

# funnel(df_raw)

import plotly.express as px

# Select top-3 regions with most site interaction
top3 = df_funnel.groupby('region')['count'].sum().nlargest(3)
top3_names = top3.index.tolist()

# Create a funnel plot
fig = px.funnel(df_funnel[df_funnel['region'].isin(top3_names)], x='count', y='event', color='region')
fig


# What are the items purchased?
df_raw[df_raw['event_name']=='add_to_cart']['page_title'].value_counts().nlargest(5)


## Network plotting

# Count up the number of transitions for graphing
df_unlogged = df2[~df2['logged_in']].groupby('transition').size().reset_index(name='weight')
df_unlogged[['source', 'target']] = df_unlogged['transition'].str.split(' -> ', expand=True)
df_unlogged

df_logged = df2[df2['logged_in']].groupby('transition').size().reset_index(name='weight')
df_logged[['source', 'target']] = df_logged['transition'].str.split(' -> ', expand=True)
df_logged

# Create graphs
G1 = nx.from_pandas_edgelist(df_unlogged, source='source', target='target', edge_attr='weight', create_using=nx.DiGraph())
G2 = nx.from_pandas_edgelist(df_logged, source='source', target='target', edge_attr='weight', create_using=nx.DiGraph())

# Edge weights
weights1 = [0.02 * G1[i][v]['weight'] for i, v in G1.edges]
weights2 = [0.02 * G2[i][v]['weight'] for i, v in G2.edges]

# Draw graph with edge weights corresponding to count of transitions

# Users NOT logged in
nx.draw_circular(G1, with_labels=True, edge_color='lightblue', node_color='lightgreen', node_size=1000, width=weights1, arrowsize=10)
plt.title('Transition Graph for Unlogged Users')
plt.savefig('transitions_unlogged.png')

# Users logged in
nx.draw_circular(G2, with_labels=True, edge_color='lightblue', node_color='lightgreen', node_size=1000, width=weights2, arrowsize=10)
plt.title('Transition Graph for Logged In Users')
plt.savefig('transitions_logged.png')


# Convert to list of lists for fitting transition matrix
# sequences = df_dummy[['event_name_int', 'event_next_int']].values.tolist()
# sequences


# NB: Got this from google AI
def fit_transition_matrix(sequences):
    """
    Estimates the transition matrix from a list of state sequences.

    Args:
        sequences: A list of lists, where each inner list represents a sequence of states.

    Returns:
        A numpy array representing the estimated transition matrix.
    """
    num_states = max(max(seq) for seq in sequences) + 1
    transition_matrix = np.zeros((num_states, num_states))

    for seq in sequences:
        for i in range(len(seq) - 1):
            current_state = seq[i]
            next_state = seq[i + 1]
            transition_matrix[current_state, next_state] += 1

    # Normalize the rows to get probabilities
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

    return transition_matrix


transition_matrix = fit_transition_matrix(sequences)
print(transition_matrix)

# Plot a heatmap

# Create a list of event names for the axes
event_names = list(text_to_int.keys())

# Update the heatmap with the event names as axis labels
sns.heatmap(transition_matrix, annot=True, cmap='YlGnBu', xticklabels=event_names, yticklabels=event_names)
plt.title('Transition Matrix Heatmap')
plt.xlabel('Next Event')
plt.ylabel('Current Event')

# Preliminary insights:

# After starting a session, 59% of new users leave the site, while 40% view an item
# After viewing an item, 20% add to cart
# Once in cart, users are going back to view item..? (confusing)
# 55% purchase once beginning checkout (pretty good, low abandon rate)
# after purchase, users end the session. (maybe is there a way to ... keep them around? discount?)
# Quite a few users go from add to cart back to viewing items (I wonder if a popup to checkout, like the
# Bonobos website does, might help convert?)

# Look at which pages people are viewing the most
df2[df2['event_name']=='view_item']['page_title'].value_counts()

# Top viewed is Sunny Side Coffee Blend

