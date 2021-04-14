import networkx as nx
from typing import List, Callable, DefaultDict, Dict
from collections import defaultdict
import numpy as np
import scipy.stats as st
aggregation_mapping = {'average': np.average, 'variance': np.var, 'skewness': st.skew,
                       'kurtosis': st.kurtosis}


def agg_local_features(node_feats: Dict, agg_type: str) -> np.float:
    return aggregation_mapping[agg_type](dict(node_feats).values())


def aggregate_features(graphs: List[nx.Graph], local_func: List[Callable],
                       agg_type: str = 'average') -> DefaultDict:
    feat_as_dic = defaultdict(list)
    for func in local_func:
        for g in graphs:
            feat_as_dic[f'{func.__name__}_{agg_type}'].append(agg_local_features(func(g), agg_type))
    return feat_as_dic


def global_features(graphs: List[nx.Graph], global_func: List[Callable]) -> DefaultDict:
    feat_as_dic = defaultdict(list)
    for func in global_func:
        for g in graphs:
            feat_as_dic[func.__name__].append(func(g))
    return feat_as_dic


def main_global_features(graphs: List[nx.Graph]):
    global_funcs = [nx.density, nx.betweenness_centrality, nx.eigenvector_centrality,
                    nx.katz_centrality_numpy, ]
    local_features = [nx.degree, nx.degree_centrality, nx.closeness_centrality, ]