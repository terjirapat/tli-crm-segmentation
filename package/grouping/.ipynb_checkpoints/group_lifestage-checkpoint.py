import pandas as pd
from package.utils import timer, load_data

def _get_lifestage(data):
    proxy = data.copy()
    proxy = proxy.loc[proxy['current_age']>=0, :]
    proxy['group_lifestage'] = None
    # proxy.loc[proxy['current_age']>=0, 'group_lifestage'] = '0-20'
    # proxy.loc[proxy['current_age']>=21, 'group_lifestage'] = '21-35'
    # proxy.loc[proxy['current_age']>=36, 'group_lifestage'] = '36-50'
    # proxy.loc[proxy['current_age']>=51, 'group_lifestage'] = '51-60'
    # proxy.loc[proxy['current_age']>=61, 'group_lifestage'] = '61+'
    proxy.loc[proxy['current_age']>=0, 'group_lifestage'] = '0-5'
    proxy.loc[proxy['current_age']>=6, 'group_lifestage'] = '6-20'
    proxy.loc[proxy['current_age']>=21, 'group_lifestage'] = '21-25'
    proxy.loc[proxy['current_age']>=26, 'group_lifestage'] = '26-35'
    proxy.loc[proxy['current_age']>=36, 'group_lifestage'] = '36-50'
    proxy.loc[proxy['current_age']>=51, 'group_lifestage'] = '51-60'
    proxy.loc[proxy['current_age']>=61, 'group_lifestage'] = '60+'
    return proxy

@timer
def group_lifestage(data, customer_path):
    proxy = data.copy()
    dtypes = {'party_rk':str}
    customer_df = load_data(customer_path, dtype=dtypes)
    customer_df = _get_lifestage(data=customer_df)
    return pd.merge(left=proxy, right=customer_df[['party_rk', 'group_lifestage']], on='party_rk', how='inner')