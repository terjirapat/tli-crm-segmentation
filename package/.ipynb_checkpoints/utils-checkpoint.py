import time 
import argparse
import pandas as pd
import yaml
from datetime import datetime

class DotDict:
    """a utility class helps converting python dictionary in to class with dot attribute
    
    Args:
        input_dict (dict) : the input dictionary
        
    Examples:
        >>> dotdict = DotDict(input_dict={'a', 1, 'b', 2})
        >>> dotdict.a
        1
    """
    def __init__(self, input_dict:dict):
        self.__dict__.update(input_dict)
        
def timer(func):
    """a wrapper function for timeing the execution time. It can be used as python decorator
    
    Examples:
        >>> timer_fn = timer(lambda x: x+1)
        >>> timer_fn(1)
        function: <lambda> is starting...
        function: <lambda> successfully executed 0.002s
        2
    """
    def wrap(*args, **kwargs):
        start = time.time()
        print(f"function: {func.__name__} is starting...", ) 
        result = func(*args, **kwargs)
        end = time.time()
        print(f"function: {func.__name__} successfully executed at {end-start}s", ) 
        return result 
    return wrap

@timer
def get_config(config_path:str) -> DotDict:
    """convert a ymal file into DoTDict, a dot notation
    Args:
        config_path (str) : a path of configuration file
    Returns:
        DotDict : a dot notation
    """
    with open(config_path, 'r') as f:
        conf = DotDict(input_dict=yaml.safe_load(f))
    return conf

@timer
def get_parameters(batch_size:int=1000, 
                   start:int=0, 
                   end:int=2000,
                   n_jobs:int=1
                  ):
    """this function will be used in final script for getting new input arguments
    
    Args:
        batch_size (int) : the size of data that will be used in each batch
        start (int) : the start index using in df.iloc[start_index:end_index,:] for filtering purpose
        end (int) : the end index using in df.iloc[start_index:end_index,:] for filtering purpose
        n_jobs (int) : the number of cpu cores will be used in this program
    Returns:
        argparse.Namespace : an object containing all input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--start', type=int, default=start)
    parser.add_argument('--end', type=int, default=end)
    parser.add_argument('--n_jobs', type=int, default=n_jobs)
    args, _ = parser.parse_known_args()
    return args

@timer
def save_file(df:pd.core.frame.DataFrame, path:str, method:str='csv')->None:
    """a function for saving pd.core.frame.DataFrame
    
    Args:
        df (pd.core.frame.DataFrame) : input dataframe
        path (str) : the save location
        method (str) : method of saving. now it has only method='csv'
    """
    if method=='csv':
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Method {method} is not implemented. Choose csv instead.")
    print(f"saved successfully at {path}")

@timer
def profile_data(data:pd.core.frame.DataFrame, path, sample=5) -> None:
    """profile data
    Args:
        data () : a dataset
        path (str) : a path for saving file
        sample (int) : a sample of each feature
    """
    df = pd.DataFrame(data.isna().sum(), columns=['missing'])
    df['missing%'] = data.isna().sum()/data.shape[0]
    df['nunique'] = data.nunique()
    df['sample'] = [data[col].unique()[:sample] for col in data.columns]
    df = df.reset_index(names='features')
    df.insert(1, 'dtype', data.dtypes.values)
    df.insert(1, 'description', '')
    df.insert(1, 'feature_group', '')
    df.insert(1, 'type', '')
    save_file(df=df, path=path, method='csv')
    
@timer
def load_data(path:str, dtype:dict=None) -> pd.core.frame.DataFrame:
    """load a csv dataset
    Args:
        path (str) : a path for csv file
    Returns:
        pd.core.frame.DataFrame : a dataset
    """
    print(f"loading data from: {path}")
    data = pd.read_csv(path, dtype=dtype)
    # data.columns = data.columns.str.upper()
    print(f"Succesfully loaded data from: {path}")
    return data

@timer
def merge_data(left:pd.core.frame.DataFrame, right:pd.core.frame.DataFrame, how:str, left_on:list, right_on:list)->pd.core.frame.DataFrame:
    """merge between two datasets and print out data shape before & after merging
    Args:
        left (pd.core.frame.DataFrame) : a base dataset
        right (pd.core.frame.DataFrame) : a dataset to join
        how (str) : joining methods including left, right, inner, outer
        left_on (list) : a list of reference keys from left dataset
        right_on (list) : a list of reference keys from right dataset
    Returns:
        pd.core.frame.DataFrame : a joined dataset
    """
    print(f"left shape: {left.shape}")
    print(f"right shape: {right.shape}")
    merged = left.merge(right=right, how='left', left_on=left_on, right_on=right_on)
    print(f"merged dataset shape: {merged.shape}")
    return merged

@timer
def convert_datetime(data:pd.core.frame.DataFrame, columns:list, format:str)->pd.core.frame.DataFrame:
    """convert datatype from string into datetime
    Args:
        data (pd.core.frame.DataFrame) : a dataset
        columns (list) : target columns that will be converted
        format (str) : a string format of the target columns
    Returns:
        pd.core.frame.DataFrame : a dataset with datetime columns
    """
    proxy = data.copy()
    for col in columns:
        proxy[col] = pd.to_datetime(proxy[col], format=format)
    return proxy

# def get_datetime(data:pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
#     """
#     Args:
#         data (pd.core.frame.DataFrame) : the input dataframe
#     Returns:
#         pd.core.frame.DataFrame : the input dataframe with column today datetime
#     """
#     data['predict_datetime'] =  datetime.now()
#     return data

def get_datetime():
    return datetime.now().strftime("%m-%Y")