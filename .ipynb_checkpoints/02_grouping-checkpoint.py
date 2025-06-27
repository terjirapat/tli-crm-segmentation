INPUT_PATH = "/opt/ml/processing/input"
INPUT_PROCESSED_PATH = f'{INPUT_PATH}/processed'
INPUT_RAW_PATH = f"{INPUT_PATH}/raw"
PACKAGE_PATH = f"{INPUT_PATH}/code/package"
CONFIG_PATH = f"{PACKAGE_PATH}/grouping/grouping.config.yaml"

OUTPUT_PATH = "/opt/ml/processing/output"
OUTPUT_FEATURE_PATH = f"{OUTPUT_PATH}/feature"
OUTPUT_SCALER_PATH = f"{OUTPUT_PATH}/scaler"

INPUT_FEATURE_PATH = f"{INPUT_PROCESSED_PATH}/feature.csv"
INPUT_FAMILY_PATH = f"{INPUT_PROCESSED_PATH}/family.csv"

INPUT_CUSTOMER_PATH = f"{INPUT_RAW_PATH}/tay_ds_customer_profile_202505091441.csv"

# -------------------------------------------------------------------------- #

import os

print(f"current dir: {os.listdir()}")
print(f"package dir: {os.listdir(PACKAGE_PATH)}")
print(f"input processed dir: {os.listdir(INPUT_PATH)}")
print(f"input raw dir: {os.listdir(INPUT_RAW_PATH)}")

# -------------------------------------------------------------------------- #

os.system(f"pip install -r {PACKAGE_PATH}/requirements.txt")

# import sklearn
# import pandas as pd

# print(f'sklearn: {sklearn.__version__}')
# print(f'pandas: {pd.__version__}')

# -------------------------------------------------------------------------- #

from package.grouping.group_behavior import group_behavior
from package.grouping.group_family import group_family
from package.grouping.group_lifestage import group_lifestage
from package.grouping.group_rider import group_rider
from package.utils import save_file, timer, load_data
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
import time

from package.utils import get_config
conf = get_config(config_path=CONFIG_PATH)

# -------------------------------------------------------------------------- #

def norm_feature(data):
    scaler = StandardScaler()
    feature_norm = scaler.fit_transform(data)
    feature_norm_df = pd.DataFrame(data=feature_norm, columns=data.columns, index=data.index)
    return scaler, feature_norm_df

# -------------------------------------------------------------------------- #

@timer
def main(conf):
    dtypes = {'party_rk':str}
    proxy = load_data(path=INPUT_FEATURE_PATH, dtype=dtypes)

    proxy = group_rider(data=proxy)
    proxy = group_family(data=proxy, family_path=INPUT_FAMILY_PATH)
    proxy = group_behavior(data=proxy)
    proxy = group_lifestage(data=proxy, customer_path=INPUT_CUSTOMER_PATH)
    
    proxy = proxy.set_index(conf.PARTY_ID_COLUMN)
    group_main = proxy[conf.GROUP_INPUT[0]]
    for group in conf.GROUP_INPUT[1:]:
        group_main = group_main + '_' + proxy[group]
    proxy['group_main'] = group_main
    save_file(proxy.reset_index(), f'{OUTPUT_FEATURE_PATH}/feature_main.csv')

    for group in sorted(proxy['group_main'].unique()):
        mask = proxy['group_main']==group
        group_df = proxy.loc[mask, :].copy()

        scaler, feature_norm_df = norm_feature(data=group_df[conf.FEATURE_INPUT])

        save_file(group_df.reset_index(), f'{OUTPUT_FEATURE_PATH}/feature{group}.csv')
        save_file(feature_norm_df.reset_index(), f'{OUTPUT_FEATURE_PATH}/feature_normalized{group}.csv')

        with open(f'{OUTPUT_SCALER_PATH}/scaler_{group}.pkl', 'wb') as f:
            pickle.dump(scaler, f)
            
    print(f"output dir: {os.listdir(OUTPUT_PATH)}")
    
if __name__=='__main__':
    main(conf)
    print("Successfully clustering")