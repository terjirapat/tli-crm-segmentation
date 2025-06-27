import pandas as pd
from package.utils import timer

@timer
def predict_model(model, method, X):
    proxy = X.copy()
    if method=='gmm':
        return predict_gmm(model=model, X=proxy)
    else:
        return predict_all(model=model, X=proxy)

def predict_all(model, X):
    labels = model.labels_
    return labels

def predict_gmm(model, X):
    labels = model.predict(X)
    return labels

def add_labels(X, labels):
    proxy = X.copy()
    proxy = proxy.reset_index()
    proxy = proxy[[proxy.columns[0]]]
    proxy['labels'] = labels
    return proxy