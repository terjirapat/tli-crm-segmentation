import pandas as pd
from package.utils import timer

def filter_active_rider(data, conf):
    return data[data['rider_status'].isin(conf.RIDER_ACTIVE_STATUS)].copy()

def _prep_feature_rider(data):
    proxy = data.copy()
    proxy = proxy.groupby('party_rk').agg(
        policy_count=('policy_no', 'count'),
        total_premium=('anp', 'sum'),
    )
    return proxy

# @timer
# def prep_feature_rider(data, conf):
#     feature_df = None
#     for key in conf.RIDER_GROUP:
#         mask = data['rider_group'].isin(conf.RIDER_GROUP[key])
#         if key=='accident':
#             mask |= data['rider_cd']=='ME' # add rider code 'ME' from 'Industrial' rider group
#             proxy = data[mask].copy()
#         else:
#             proxy = data[mask].copy()
#         proxy = _prep_feature_rider(data=proxy)
#         proxy.columns = proxy.columns + f'_{key}'
#         feature_df = pd.concat(objs=[feature_df, proxy], axis=1)
#     return feature_df

@timer
def prep_feature_rider(data, conf):
    feature_df = None
    for key in conf.RIDER_GROUP:
        mask = data['rider_group'].isin(conf.RIDER_GROUP[key])
        proxy = data[mask].copy()
        proxy = _prep_feature_rider(data=proxy)
        proxy.columns = proxy.columns + f'_{key}'
        feature_df = pd.concat(objs=[feature_df, proxy], axis=1)
    return feature_df

@timer
def get_partyrk_rider(policy_df, rider_df):
    proxy = policy_df[['party_rk', 'policy_no', 'payment_mode', 'plan_index']].copy()
    mask = proxy['payment_mode']!=9 # remove single premium policy becase single premium policy do not have rider
    mask &= ~proxy['plan_index'].isin(['PA', 'TERM', 'Term']) # ['PA', 'TERM', 'Term'] can not buy rider so if there is rider with these plan type the data is incorrect
    proxy = proxy[mask]
    proxy = pd.merge(left=proxy, right=rider_df, how='inner', on='policy_no')
    return proxy