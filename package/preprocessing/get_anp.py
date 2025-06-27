import pandas as pd
from package.utils import timer

@timer
def get_premium_baseplan(data):
    proxy = data.copy()
    proxy['anp'] = proxy['life_prem'].fillna(0) + proxy['rsp_life_prem'].fillna(0) + proxy['life_extra_prem'].fillna(0)
    return proxy

@timer
def get_premium_rider(data):
    proxy = data.copy()
    proxy['anp'] = proxy['premium'].fillna(0) + proxy['extrapremium'].fillna(0)
    return proxy

@timer
def get_anp(data):
    proxy = data.copy()
    proxy.loc[proxy['payment_mode']==0, 'anp'] = proxy.loc[proxy['payment_mode']==0, 'anp'] * 12
    proxy.loc[proxy['payment_mode']==2, 'anp'] = proxy.loc[proxy['payment_mode']==2, 'anp'] * 2
    proxy.loc[proxy['payment_mode']==4, 'anp'] = proxy.loc[proxy['payment_mode']==4, 'anp'] * 4
    # siglepremium no anp, edit later
    proxy.loc[proxy['payment_mode']==9, 'anp'] = 0
    return proxy