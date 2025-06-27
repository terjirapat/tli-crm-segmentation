import pandas as pd
from package.utils import timer

def _get_lastpolicy(data):
    data['effective_dt'] = pd.to_datetime(data['effective_dt'])
    last_effective_dt = data.groupby('party_rk')['effective_dt'].max().reset_index()
    proxy = pd.merge(data, last_effective_dt, on=['party_rk', 'effective_dt'], how='inner')
    return proxy

@timer
def prep_feature_beneficiary(data, conf):
    proxy = _get_lastpolicy(data=data)
    proxy = data.groupby('party_rk').sum(conf.BENEFICIARY_INPUT)[conf.BENEFICIARY_INPUT]
    return proxy.div(proxy.sum(axis=1), axis=0).fillna(0)