import pandas as pd
import json
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Any, Literal
import typing

@dataclass
class Log:
    df:pd.DataFrame
    model_name:Literal["kmeans", "gmm", "dbscan"]
    parameter:Dict[str, Any]
    path:str
    timestamp:int
    customer_group:str
    
    def __post_init__(self):
        self.shape = str(self.df.shape)
        self.features = str(self.df.columns.tolist())
        self.active = False
        self.df = None
        self.parameter = json.dumps(self.parameter)
        available_models = self.__annotations__['model_name'].__args__
        if self.model_name not in available_models:
            raise ValueError(f"{self.model_name} is not available, try {available_models} instead.")
    def save_to_json(self, path="./log.json"):
        json_dict = self.__dict__.copy()
        json_dict.pop("df")
        with open(path, "w") as f:
            # json.dump(list(json_dict), f)
            json.dump(json_dict, f)
    def info(self):
        return self.__dict__