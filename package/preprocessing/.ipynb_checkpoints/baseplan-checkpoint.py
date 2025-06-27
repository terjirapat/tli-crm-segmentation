import pandas as pd
from package.utils import timer

def filter_active_policy(data, conf):
    proxy = data.copy()
    proxy = proxy[proxy['policy_status1'].isin(conf.POLICY_ACTIVE_STATUS)]
    # remove customer that death, disabled person, ฟอกเงิน
    mask = proxy['policy_status1'].isin(conf.POLICY_REMOVE_STATUS)
    mask |= proxy['policy_status2'].isin(conf.POLICY_REMOVE_STATUS)
    remove_list = proxy.loc[mask, 'party_rk']
    mask = ~proxy['party_rk'].isin(remove_list)
    proxy = proxy[mask]
    return proxy

def _prep_feature_policy(data):
    proxy = data.copy()
    proxy = proxy.groupby('party_rk').agg(
        policy_count=('policy_no', 'nunique'),
        total_premium=('anp', 'sum'),
    )
    return proxy

@timer
def prep_feature_policy(data, conf):
    feature_df = None
    for key in conf.BASEPLAN_GROUP:
        proxy = data[data['plan_index'].isin(conf.BASEPLAN_GROUP[key])].copy()
        proxy = _prep_feature_policy(data=proxy)
        proxy.columns = proxy.columns + f'_{key}'
        feature_df = pd.concat(objs=[feature_df, proxy], axis=1)
    return feature_df