import pandas as pd
from package.utils import timer, load_data

@timer
def createfamilyflag_pipeline(policy_df, policybene_path, beneid_path):
    dtypes = {'policyno':str, 'party_id_name_bd':str, 'party_id_govt':str, 'policy_govt':str, 'party_id':str, 'dup_party_id_govt':str,}
    policybene_df = load_data(policybene_path, dtype=dtypes)
    beneid_df = load_data(beneid_path, dtype=dtypes)
    
    # get active policyno
    policyactive_list = set(policy_df['policy_no'])
    
    # get active beneficiary partyid
    mask = policybene_df['policy_govt'].isin(policyactive_list)
    activebeneid_list = set(policybene_df.loc[mask, 'party_id'])
    
    # get policy that having active beneficiary
    beneid_df.loc[beneid_df['party_id_govt'].isna(), 'party_id_govt'] = beneid_df.loc[beneid_df['party_id_govt'].isna(), 'party_id_name_bd']
    mask = ~beneid_df['party_id_govt'].isna()
    mask &= beneid_df['relationshipcode'].isin([1, 2, 3, 4, 5, 22, 43, 44])
    mask &= beneid_df['policyno'].isin(policyactive_list)
    mask &= beneid_df['party_id_govt'].isin(activebeneid_list)
    beneid_df = beneid_df.loc[mask, :]
    familypolicyno_list = set(beneid_df['policyno'])
    print(len(familypolicyno_list))
    print(beneid_df.info())
    
    #### check active policy on duplicated partyid
    mask = beneid_df.duplicated(subset='party_id_govt')
    duppolicyno_list = set(beneid_df[mask]['policyno'])
    print(beneid_df[mask])
    print(f'no of active policy on duplicate partyid:{len(duppolicyno_list)}')
    ####
    
    # get df having party_rk and family flag
    proxy = policy_df[['party_rk', 'policy_no']].copy()
    proxy['is_family'] = 0
    proxy.loc[proxy['policy_no'].isin(familypolicyno_list), 'is_family'] = 1
    proxy = proxy[['party_rk', 'is_family']].sort_values(by='is_family', ascending=False)
    proxy = proxy.drop_duplicates(subset='party_rk', keep='first')
    print(proxy['is_family'].value_counts())
    return proxy
    