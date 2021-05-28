import networkx as nx
from typing import List, Callable, DefaultDict, Dict, NoReturn
from collections import defaultdict
import numpy as np
from collections import ChainMap
import torch
from karateclub.graph_embedding import Graph2Vec, FGSD
from netlsd import heat, wave
import os
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


def padding_by_zeros(f: List[Dict], dict_to_pad: Dict, padding_name, zeros_size: int) -> NoReturn:
    max_size = max([len(list(global_feat.values())) for global_feat in f])
    for i in range(max_size):
        dict_to_pad[f'{padding_name}_{i}'] = [0] * zeros_size


def features_by_values(graphs: List[nx.Graph], func_features: List[Callable]) -> DefaultDict:
    feat_as_dict = defaultdict(list)
    for func in tqdm(func_features):
        global_feature_vals_lst = [func(g) for g in graphs]
        degrees = [nx.degree(g) for g in graphs]
        padding_by_zeros(global_feature_vals_lst, feat_as_dict, func.__name__, len(graphs))

        for j, global_feature_vals in enumerate(global_feature_vals_lst):
            for i, val in enumerate(global_feature_vals.values()):
                # Todo: Change it to more readable
                if func.__name__ == 'average_neighbor_degree':
                    anatomical_label = graphs[j].nodes()[i]['label']
                    feat_as_dict[f'{func.__name__}_{i}_deg({degrees[j][i]})_label_{anatomical_label}'][j] = val
                else:
                    feat_as_dict[f'{func.__name__}_{i}'][j] = val
    return feat_as_dict


def measurement_for_unconnected_global(func: Callable) -> Callable:
    def res_func(graph):
        return np.average([func(graph.subgraph(comp)) for comp in nx.connected_components(graph)])
    res_func.__name__ = func.__name__
    return res_func


def measurements_for_unconnected_local(func: Callable) -> Callable:
    def res_func(graph):
        res = {k: v for x in [func(graph.subgraph(comp)) for comp in nx.connected_components(graph)]
                                  for k, v in x.items()}
        return res
    res_func.__name__ = func.__name__
    return res_func


def main_global_features(graphs: List[nx.Graph]) -> pd.DataFrame:
    # nx.number_weakly_connected_components nx.number_attracting_components - for directed, nx.number_strongly_connected_components,
    # nx.node_connectivity,# nx.graph_number_of_cliques,#nx.local_efficiency, # nx.sigma, nx.omega,
    # bipartite graph latapy_clustering  node_redundancy
    # time running nx.current_flow_betweenness_centrality, measurement_for_unconnected_global(nx.wiener_index)

    global_funcs = [nx.density, nx.number_connected_components, nx.average_clustering,
                    nx.degree_assortativity_coefficient, nx.degree_pearson_correlation_coefficient,
                    spectral_bipartivity, nx.global_reaching_centrality,
                    edge_connectivity, measurement_for_unconnected_global(nx.diameter), nx.global_efficiency,
                    nx.number_of_isolates, nx.overall_reciprocity]

    global_feat = global_features(graphs, global_funcs)

    def rich_club_func(graph):
        return nx.rich_club_coefficient(graph, normalized=False)
    rich_club_func.__name__ = nx.rich_club_coefficient.__name__


    others = [nx.average_neighbor_degree, nx.average_degree_connectivity,
              rich_club_func]

    others_feat = features_by_values(graphs, others)
    local_features = [nx.clustering, nx.degree, nx.degree_centrality, nx.closeness_centrality,
                      nx.betweenness_centrality, #lambda g: nx.eigenvector_centrality(g, max_iter=1000),
                      nx.katz_centrality_numpy, nx.communicability_betweenness_centrality,
                      nx.harmonic_centrality, nx.load_centrality, nx.subgraph_centrality, nx.triangles,
                      nx.pagerank_numpy, measurements_for_unconnected_local(nx.second_order_centrality),
                      measurements_for_unconnected_local(nx.eccentricity)]
    res = {}
    # unconnected nx.approximate_current_flow_betweenness_centrality, nx.second_order_centrality, nx.eccentricity,
    res.update(aggregate_features(graphs, local_features))
    res.update(global_feat)
    res.update(others_feat)
    return pd.DataFrame(res)
