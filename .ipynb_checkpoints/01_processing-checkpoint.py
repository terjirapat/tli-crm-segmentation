INPUT_PATH = "/opt/ml/processing/input"
PACKAGE_PATH = f"{INPUT_PATH}/code/package" # in the notebook will end with "/"
CONFIG_PATH = f"{PACKAGE_PATH}/preprocessing/preprocessing.config.yaml"

OUTPUT_PATH = "/opt/ml/processing/output"
# OUTPUT_FEATURE_PATH = f"{OUTPUT_PATH}/feature"
# OUTPUT_SCALER_PATH = f"{OUTPUT_PATH}/scaler"

INPUT_RAW_PATH = f"{INPUT_PATH}/raw"
INPUT_POLICY_PATH = f"{INPUT_RAW_PATH}/ds_mst_policy_profile_202505091742.csv"
INPUT_RIDER_PATH = f"{INPUT_RAW_PATH}/tay_ds_mst_rider_profile_202504090908.csv"
INPUT_BENEID_PATH = f"{INPUT_RAW_PATH}/stg_tb_mstperson_beneficiary_party_id_name_bd_party_id_govt_202505251050.csv"
INPUT_POLICYBENE_PATH = f"{INPUT_RAW_PATH}/stg_tb_policy_policyparticipant_202505220106.csv"

INPUT_MODEL_PATH = f"{INPUT_PATH}/model"
INPUT_BENEMODEL_PATH = f"{INPUT_MODEL_PATH}/bene_model.pkl"

# -------------------------------------------------------------------------- #

import os
os.system(f"pip install -r {PACKAGE_PATH}/requirements.txt")

# import sklearn
# import pandas as pd

# print(f'sklearn: {sklearn.__version__}')
# print(f'pandas: {pd.__version__}')

# -------------------------------------------------------------------------- #

from package.preprocessing.baseplan import filter_active_policy, prep_feature_policy
from package.preprocessing.rider import filter_active_rider, prep_feature_rider, get_partyrk_rider
from package.preprocessing.beneficiary import prep_feature_beneficiary
from package.preprocessing.get_anp import get_premium_baseplan, get_premium_rider, get_anp
from package.preprocessing.driver import impute_0anp_driver, split_onlytermpa_driver, get_lookup_bin, get_feature_bin
from package.preprocessing.family import createfamilyflag_pipeline
from package.utils import save_file, timer, load_data
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
import time

from package.utils import get_config
conf = get_config(config_path=CONFIG_PATH)

# -------------------------------------------------------------------------- #

print(f"current dir: {os.listdir()}")
print(f"input dir: {os.listdir(INPUT_PATH)}")
print(f"input raw dir: {os.listdir(INPUT_RAW_PATH)}")
print(f"input model dir: {os.listdir(INPUT_MODEL_PATH)}")

# -------------------------------------------------------------------------- #

@timer
def main(conf):
    dtypes = {'policy_no':str, 'party_rk':str}
    policy_df = load_data(path=INPUT_POLICY_PATH, dtype=dtypes)

    # base plan
    policy_df = filter_active_policy(data=policy_df, conf=conf)
    policy_df = get_premium_baseplan(data=policy_df)
    policy_df = get_anp(data=policy_df)
    policy_df = impute_0anp_driver(data=policy_df)
    feature_df = prep_feature_policy(data=policy_df, conf=conf)
    
    # family
    family_df = createfamilyflag_pipeline(policy_df=policy_df, policybene_path=INPUT_POLICYBENE_PATH, beneid_path=INPUT_BENEID_PATH)

    # beneficiary
    proxy = prep_feature_beneficiary(data=policy_df, conf=conf)
    feature_df = pd.merge(left=feature_df, right=proxy, how='left', left_index=True, right_index=True)

    # rider
    rider_df = load_data(path=INPUT_RIDER_PATH, dtype=dtypes)
    rider_df = filter_active_rider(data=rider_df, conf=conf)
    proxy = get_partyrk_rider(policy_df=policy_df, rider_df=rider_df)
    proxy = get_premium_rider(data=proxy)
    proxy = get_anp(data=proxy)
    proxy = prep_feature_rider(data=proxy, conf=conf)
    feature_df = pd.merge(left=feature_df, right=proxy, how='left', left_index=True, right_index=True)
    
    # other feature
    feature_df = feature_df.fillna(0)
    feature_df['total_premium'] = feature_df.filter(like='total_premium').sum(axis=1)
    
    # split pa term
    feature_df, termpa_df = split_onlytermpa_driver(data=feature_df)
    
    # get bins premium
    bin_df = None
    for col in conf.PREMIUM_INPUT:
        lookup_df = get_lookup_bin(data=feature_df, column=col, k=4)
        feature_df = get_feature_bin(data=feature_df, lookup=lookup_df, column=col)
        
        proxy = feature_df.groupby(f'bin_{col}').agg(min=(col, 'min'), max=(col, 'max'), count=(col, 'count')).sort_values(by='min').reset_index()
        proxy = proxy.rename(columns={f'bin_{col}':'bin'})
        proxy['feature'] = col
        bin_df = pd.concat([bin_df, proxy], axis=0)
    save_file(bin_df, f'{OUTPUT_PATH}/bin_range.csv')
    
    # bene group
    with open(INPUT_BENEMODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    feature_df['group_bene'] = model.predict(feature_df[conf.BENEFICIARY_INPUT])
    feature_df['group_bene'] = feature_df['group_bene'].map(conf.BENEFICIARY_GROUP)
    feature_df = pd.get_dummies(feature_df, columns=['group_bene'], dtype='int8')
    # feature_df = feature_df.drop(columns=['group_bene_nofamily'])
    
    feature_df.index.name = 'party_rk'

    save_file(feature_df.reset_index(), f'{OUTPUT_PATH}/feature.csv')
    save_file(termpa_df.reset_index(), f'{OUTPUT_PATH}/feature_termpa.csv')
    save_file(family_df, f'{OUTPUT_PATH}/family.csv')

    print(f"output dir: {os.listdir(OUTPUT_PATH)}")

if __name__=='__main__':
    main(conf)
    print("Successfully clustering")