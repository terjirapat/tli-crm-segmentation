from package.utils import timer, load_data

@timer
def group_family(data, family_path):
    dtypes = {'party_rk':str}
    family_df = load_data(family_path, dtype=dtypes)
    mask = family_df['is_family']==1
    isfamily_list = set(family_df.loc[mask, 'party_rk'])
    
    proxy = data.copy()
    proxy['group_family'] = None
    mask = proxy['party_rk'].isin(isfamily_list)
    proxy.loc[mask, 'group_family'] = 'family'
    proxy['group_family'] = proxy['group_family'].fillna('nofamily')
    return proxy