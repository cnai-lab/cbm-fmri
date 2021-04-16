import networkx as nx
from typing import List, Callable, DefaultDict, Dict
from collections import defaultdict
import numpy as np
from karateclub.graph_embedding import  Graph2Vec, FGSD
from netlsd import heat, wave
from networkx.algorithms.bipartite import *
from networkx.algorithms import *

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


def main_global_features(graphs: List[nx.Graph]):

    global_funcs = [nx.density, nx.betweenness_centrality, nx.number_connected_components, nx.average_clustering,
                    nx.degree_assortativity_coefficient, nx.degree_pearson_correlation_coefficient,
                    nx.node_connectivity, spectral_bipartivity, nx.group_betweenness_centrality,
                    nx.group_closeness_centrality, nx.global_reaching_centrality, nx.graph_number_of_cliques,
                    nx.number_weakly_connected_components, nx.number_attracting_components,
                    nx.number_strongly_connected_components, connectivity.node_connectivity, edge_connectivity,
                    nx.diameter, nx.local_efficiency, nx.global_efficiency, nx.maximum_flow_value,
                    nx.number_of_isolates, nx.overall_reciprocity, nx.sigma, nx.omega, nx.s_metric]

    global_feat = global_features(graphs, global_funcs)

    others = [nx.average_neighbor_degree, nx.average_degree_connectivity, nx.k_nearest_neighbors,
              nx.rich_club_coefficient]

    local_features = [nx.clustering,  latapy_clustering, nx.degree, nx.degree_centrality,
                      nx.closeness_centrality, node_redundancy, nx.betweenness_centrality, nx.eigenvector_centrality,
                      nx.katz_centrality_numpy, nx.closeness_centrality, nx.incremental_closeness_centrality,
                      nx.current_flow_betweenness_centrality, nx.information_centrality,
                      nx.approximate_current_flow_betweenness_centrality, nx.communicability_betweenness_centrality,
                      nx.harmonic_centrality, nx.load_centrality, nx.subgraph_centrality, nx.dispersion,
                      nx.percolation_centrality, nx.second_order_centrality, nx.trophic_levels, nx.triangles,
                      nx.square_clustering, nx.communicability, connectivity.local_node_connectivity,
                      connectivity.local_edge_connectivity, nx.eccentricity, nx.pagerank_numpy, nx.wiener_index]
