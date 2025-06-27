from package.utils import timer

def group_99percentile_anp(data):
    proxy = data.copy()
    mask = proxy['total_premium'] > proxy['total_premium'].quantile(0.99)
    proxy.loc[mask, 'group_behavior'] = '99pct'
    return proxy

def group_0anp(data):
    proxy = data.copy()
    mask = proxy['total_premium'] == 0
    proxy.loc[mask, 'group_behavior'] = '0anp'
    return proxy

def group_1hold(data):
    proxy = data.copy()
    policy_count = proxy['policy_count_saving'] + proxy['policy_count_wholelife']
    mask = policy_count == 1
    proxy.loc[mask, 'group_behavior'] = '1hold'
    return proxy

@timer
def group_behavior(data):
    proxy = data.copy()
    proxy['group_behavior'] = 'other'
    proxy = group_1hold(data=proxy)
    proxy = group_0anp(data=proxy)
    proxy = group_99percentile_anp(data=proxy)
    return proxy