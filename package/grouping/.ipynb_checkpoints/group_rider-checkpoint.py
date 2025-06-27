from package.utils import timer

@timer
def group_rider(data):
    proxy = data.copy()
    proxy['group_rider'] = None
    mask = (proxy['policy_count_health'] + proxy['policy_count_ci']) > 0
    proxy.loc[mask, 'group_rider'] = 'rider'
    proxy['group_rider'] = proxy['group_rider'].fillna('norider')
    return proxy