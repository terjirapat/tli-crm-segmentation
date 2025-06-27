import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from package.utils import timer

@timer
def impute_0anp_driver(data):
    proxy = data.copy()
    mask = proxy['policy_status1']!='I'
    proxy.loc[mask, 'anp'] = 0
    return proxy

@timer
def split_onlytermpa_driver(data):
    # split party_rk that hold only termpa
    proxy = data.copy().fillna(0)
    # col_list = ['policy_count_saving', 'policy_count_wholelife']
    col_list = [col for col in proxy.columns if 'count' in col and 'termpa' not in col]
    mask = proxy['policy_count_termpa']!=0
    for col in col_list:
        mask &= proxy[col]==0
    termpa_df = proxy[mask].copy()
    
    # remove party_rk that hold only termpa
    proxy = data.copy()
    col_list = [col for col in proxy.columns if 'termpa' not in col]
    mask = ~proxy.index.isin(termpa_df.index)
    feature_df = proxy.loc[mask, col_list].copy()
    return feature_df, termpa_df

# @timer
# def get_feature_bintotalpremium_driver(data):
#     proxy = data.copy()
#     mask = proxy['total_premium'] <= proxy['total_premium'].quantile(0.99)
#     mask &= proxy['total_premium'] != 0
#     proxy = proxy.loc[mask, :].copy()
#     X = proxy['total_premium']
#     X = np.array(X).reshape(-1, 1)

#     model = KMeans(n_clusters=4, random_state=0)
#     model.fit(X)
#     proxy['bin_total_premium'] = model.labels_
#     bin_df = proxy.groupby('bin_total_premium').agg(min=('total_premium', 'min'), max=('total_premium', 'max')).sort_values(by='min').reset_index(drop=True)
#     print(bin_df)
    
#     proxy = data.copy()
#     proxy['bin_total_premium'] = None

#     for i in range(len(bin_df)):
#         min_val = bin_df.loc[i, 'min']
#         max_val = bin_df.loc[i, 'max']
#         mask = proxy['total_premium']>=min_val
#         mask &= proxy['total_premium']<=max_val
#         proxy.loc[mask, 'bin_total_premium'] = i+1
#     proxy.loc[proxy['total_premium'] == 0, 'bin_total_premium'] = 0
#     proxy.loc[proxy['total_premium'] > proxy['total_premium'].quantile(0.99), 'bin_total_premium'] = proxy['bin_total_premium'].max()+1
#     return proxy

def get_lookup_bin(data, column, k):
    proxy = data.copy()
    mask = proxy[column] <= proxy[column].quantile(0.99)
    mask &= proxy[column] != 0
    proxy = proxy.loc[mask, :].copy()
    
    X = proxy[column].copy()
    X = np.array(X).reshape(-1, 1)

    model = KMeans(n_clusters=k, random_state=0, n_init='auto')
    model.fit(X)
    proxy[f'bin_{column}'] = model.labels_
    lookup_df = proxy.groupby(f'bin_{column}').agg(min=(column, 'min'), max=(column, 'max')).sort_values(by='min').reset_index(drop=True)
    print(column)
    print(lookup_df)
    return lookup_df

@timer
def get_feature_bin(data, lookup, column):
    proxy = data.copy()
    proxy[f'bin_{column}'] = None
    for i in range(len(lookup)):
        min_val = lookup.loc[i, 'min']
        max_val = lookup.loc[i, 'max']
        mask = proxy[column]>=min_val
        mask &= proxy[column]<=max_val
        proxy.loc[mask, f'bin_{column}'] = i+1
    proxy.loc[proxy[column] == 0, f'bin_{column}'] = 0
    proxy.loc[proxy[column] > proxy[column].quantile(0.99), f'bin_{column}'] = proxy[f'bin_{column}'].max()+1
    return proxy