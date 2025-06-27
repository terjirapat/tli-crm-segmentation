INPUT_PATH = "/opt/ml/processing/input"
PACKAGE_PATH = f"{INPUT_PATH}/code/package" # in the notebook will end with "/"
CONFIG_PATH = f"{PACKAGE_PATH}/clustering/clustering.config.yaml"

OUTPUT_PATH = "/opt/ml/processing/output"
OUTPUT_MODEL_PATH = f"{OUTPUT_PATH}/model"
OUTPUT_LOG_PATH = f"{OUTPUT_PATH}/log"
OUTPUT_PREDICT_PATH = f"{OUTPUT_PATH}/inferenced"

# INPUT_FEATURE_PATH = f"{INPUT_PATH}/segment_feature_normalized.csv"
# INPUT_FEATURE_PATH = f"{INPUT_PATH}/feature_normalized_99pct.csv"

# -------------------------------------------------------------------------- #

import os
os.system(f"pip install -r {PACKAGE_PATH}/requirements.txt")

# import sklearn
# import pandas as pd

# print(f'sklearn: {sklearn.__version__}')
# print(f'pandas: {pd.__version__}')

# -------------------------------------------------------------------------- #

import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument("--model-name", type=str)
parser.add_argument("--model-param", type=str)
parser.add_argument("--customer-group", type=str)

args = parser.parse_args()

model_name = args.model_name
model_param = json.loads(args.model_param)
customer_group = args.customer_group

print(f"Model Name: {model_name}")
print(f"Model Param: {model_param}")
print(f"Customer Group: {customer_group}")

# -------------------------------------------------------------------------- #

from package.clustering.get_model import get_model
from package.clustering.get_log import Log
from package.clustering.predict import predict_model, add_labels
from package.utils import save_file, timer, load_data
# import pandas as pd
import pickle
import time

from package.utils import get_config
conf = get_config(config_path=CONFIG_PATH)

# -------------------------------------------------------------------------- #

print(f"current dir: {os.listdir()}")
print(f"input dir: {os.listdir(INPUT_PATH)}")

# -------------------------------------------------------------------------- #

@timer
def train_model(model, X):
    return model.fit(X)

# -------------------------------------------------------------------------- #

@timer
def main(conf):
    X = load_data(path=f'{INPUT_PATH}/{conf.CUSTOMER_GROUP[customer_group]}')
    X = X.set_index(conf.PARTY_ID_COLUMN)

    # train and predict
    model = get_model(method=model_name, params=model_param)
    model = train_model(model=model, X=X)
    labels = predict_model(model=model, method=model_name, X=X)
    result = add_labels(X=X, labels=labels)

    # get log
    timestamp = int(time.time())
    model_log = Log(
        df=X,
        model_name=model_name,
        parameter=model_param,
        timestamp=timestamp,
        path=f"{conf.S3_MODEL_PATH}/{timestamp}/{timestamp}.pkl",
        customer_group=customer_group,
    )
    
    path = f'{OUTPUT_MODEL_PATH}/{timestamp}'
    os.makedirs(path)
    with open(f'{OUTPUT_MODEL_PATH}/{timestamp}/{timestamp}.pkl', 'wb') as f:
        pickle.dump(model, f)

    model_log.save_to_json(path=f"{OUTPUT_LOG_PATH}/{timestamp}.json")
    save_file(result, f'{OUTPUT_PREDICT_PATH}/{timestamp}.csv')

    print(f"output dir: {os.listdir(OUTPUT_PATH)}")

if __name__=='__main__':
    main(conf)
    print("Successfully clustering")