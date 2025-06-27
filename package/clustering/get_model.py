from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from package.utils import timer

@timer
def get_model(method:str, params:dict):
    if method == 'kmeans':
        return KMeans(**params)
    if method == 'gmm':
        return GaussianMixture(**params)
    if method == 'dbscan':
        return DBSCAN(**params)
    if method == 'agglo':
        return AgglomerativeClustering(**params)
    raise ValueError(f"{method} is not implemented!")