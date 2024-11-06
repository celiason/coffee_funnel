## FUNNEL PLOT

import pandas as pd

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

