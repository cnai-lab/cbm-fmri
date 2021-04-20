import networkx as nx
from typing import List, Callable, DefaultDict, Dict, NoReturn
from collections import defaultdict
import numpy as np
from karateclub.graph_embedding import  Graph2Vec, FGSD
from netlsd import heat, wave
import os
from utils import *
from networkx.algorithms.bipartite import *
from networkx.algorithms import *
import datetime
import pandas as pd
from tqdm import tqdm
import scipy.stats as st
from multiprocessing import Pool

aggregation_mapping = {'average': np.average, 'variance': np.var, 'skewness': st.skew,
                       'kurtosis': st.kurtosis}


def agg_local_features(node_feats: Dict, agg_type: str) -> np.float:
    return aggregation_mapping[agg_type](list(dict(node_feats).values()))


def aggregate_features(graphs: List[nx.Graph], local_func: List[Callable]) -> DefaultDict:
    feat_as_dic = defaultdict(list)
    for func in tqdm(local_func):
        old_time = datetime.datetime.now()
        for g in graphs:
            loc_feat = func(g)
            for agg_type in aggregation_mapping.keys():
                feat_as_dic[f'{func.__name__}_{agg_type}'].append(agg_local_features(loc_feat, agg_type))
        # write_time_of_function(func.__name__, old_time)
    return feat_as_dic


def global_features(graphs: List[nx.Graph], global_func: List[Callable]) -> DefaultDict:
    feat_as_dic = defaultdict(list)
    for func in tqdm(global_func):
        time = datetime.datetime.now()
        for g in graphs:
            feat_as_dic[func.__name__].append(func(g))
        # write_time_of_function(func.__name__, time)
    return feat_as_dic


def embedding_features(graphs: List[nx.Graph], embedding_func: Callable) -> DefaultDict:
    embed_as_dic = defaultdict(list)
    embedding_func(graphs)
    return embed_as_dic


def features_by_type(feat_type: str, graphs: List[nx.Graph], dim: int):

    def heat_embedding():
        return [heat(graph, normalized_laplacian=False, timescales=np.logspace(-2, 2, dim)) for graph in graphs]

    def wave_embedding():
        return [wave(graph, normalized_laplacian=False, timescales=np.linspace(0, 2 * np.pi, dim)) for graph in graphs]

    def fgsd_embedding():
        fgsd_model = FGSD(hist_bins=dim)
        fgsd_model.fit(graphs)
        return fgsd_model.get_embedding()

    def graph2vec():
        g2_vec_model = Graph2Vec(dimensions=dim)
        g2_vec_model.fit(graphs)
        return g2_vec_model.get_embedding()

    mapping_embed = {'heat': heat_embedding, 'wave': wave_embedding, 'fgsd': fgsd_embedding, 'graph2vec': graph2vec}
    return mapping_embed[feat_type]()


def features_by_values(graphs: List[nx.Graph], func_features: List[Callable]) -> DefaultDict:
    feat_as_dict = defaultdict(list)
    for func in tqdm(func_features):
        old_time = datetime.datetime.now()
        for g in graphs:
            global_feature_vals = func(g)
            for i, val in enumerate(global_feature_vals.values()):
                feat_as_dict[f'{func.__name__}_{i}'].append(val)
        # write_time_of_function(func.__name__, old_time)
    keys_to_remove = [key for key in feat_as_dict.keys() if len(feat_as_dict[key]) < len(graphs)]
    for key in keys_to_remove:
        feat_as_dict.pop(key)

    return feat_as_dict


    pass


def main_global_features(graphs: List[nx.Graph]) -> pd.DataFrame:
    # nx.number_weakly_connected_components nx.number_attracting_components - for directed, nx.number_strongly_connected_components,
    # nx.node_connectivity,# nx.graph_number_of_cliques,#nx.local_efficiency, # nx.sigma, nx.omega,
    # bipartite graph latapy_clustering  node_redundancy
    # time running nx.current_flow_betweenness_centrality,

    global_funcs = [nx.density, nx.betweenness_centrality, nx.number_connected_components, nx.average_clustering,
                    nx.degree_assortativity_coefficient, nx.degree_pearson_correlation_coefficient,
                    spectral_bipartivity,  nx.global_reaching_centrality,
                    edge_connectivity, nx.diameter, nx.global_efficiency,
                    nx.number_of_isolates, nx.overall_reciprocity, nx.wiener_index]

    global_feat = global_features(graphs, global_funcs)
    others = [nx.average_neighbor_degree, nx.average_degree_connectivity,
              lambda g: nx.rich_club_coefficient(g, normalized=False)]

    others_feat = features_by_values(graphs, others)
    local_features = [nx.clustering, nx.degree, nx.degree_centrality, nx.closeness_centrality,
                      nx.betweenness_centrality, nx.eigenvector_centrality, nx.katz_centrality_numpy,
                      nx.approximate_current_flow_betweenness_centrality, nx.communicability_betweenness_centrality,
                      nx.harmonic_centrality, nx.load_centrality, nx.subgraph_centrality,
                      nx.second_order_centrality, nx.triangles,
                      nx.eccentricity, nx.pagerank_numpy]
    res = {}

    res.update(aggregate_features(graphs, local_features))
    res.update(global_feat)
    res.update(others_feat)
    return pd.DataFrame(res)


