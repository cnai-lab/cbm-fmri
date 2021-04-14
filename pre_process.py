import numpy as np
from nilearn.input_data import NiftiSpheresMasker
from nilearn.connectome import ConnectivityMeasure
from typing import List, Callable, Union, Tuple, NoReturn
import networkx as nx
import os
from paths import *
import nilearn.datasets as datasets
from copy import deepcopy


def load_scans(scan_paths: List[str], data_type: str = 'correlation') -> \
        Union[List[np.ndarray], Tuple[List[np.ndarray], List[np.ndarray]]]:

    time_series_lst, corr_lst = [], []

    for path in scan_paths:
        time_series = path_to_time_series(path)
        time_series_lst.append(time_series)

    if data_type == 'time_series':
        return time_series_lst

    correlations = time_series_to_correlation(time_series_lst)
    names = [os.path.basename(path) for path in scan_paths]
    save_correlations(names, correlations)

    if data_type == 'correlation':
        return correlations

    if data_type == 'both':
        return time_series_lst, correlations

    raise ValueError('Data type should be one of the [correlation, time_series, both]')


def save_correlations(names: List[str], correlations: List[np.ndarray]) -> NoReturn:
    for name, corr in zip(names, correlations):
        path_to_save = os.path.join(SAVE_PATH, name)
        np.save(f'{path_to_save}.npy', corr)


def path_to_time_series(path: str) -> np.ndarray:

    power_atlas = datasets.fetch_coords_power_2011()
    coords = np.vstack((power_atlas.rois['x'], power_atlas.rois['y'], power_atlas.rois['z'])).T
    spheres_masker = NiftiSpheresMasker(seeds=coords, smoothing_fwhm=6, radius=5., detrend=True, standardize=True,
                                        low_pass=0.1, high_pass=0.01, t_r=2.5)
    time_series = spheres_masker.fit_transform(path)
    time_series_cleaned = np.nan_to_num(time_series)
    return time_series_cleaned


def time_series_to_correlation(time_series_lts: List[np.ndarray], is_abs: bool = False) -> List[np.ndarray]:

    connectivity_measure = ConnectivityMeasure(kind='correlation')
    corr_mat_lst = connectivity_measure.fit_transform(time_series_lts)
    corr_mat_lst_fixed = [np.fill_diagonal(corr_mat) for corr_mat in corr_mat_lst] # Are we want this?
    if is_abs:
        corr_mat_lst_fixed = [np.abs(corr_mat) for corr_mat in corr_mat_lst_fixed]
    else:
        for corr_mat in corr_mat_lst_fixed:
            corr_mat[corr_mat < 0] = 0
    return corr_mat_lst_fixed


def filter_edges(filter_type: str, graphs: List[nx.Graph], param) -> List[nx.Graph]:
    mapping_filter = {'density': filter_by_dens, 'threshold': filter_by_threshold, 'pmfg': filter_by_pmfg}
    res = []
    for graph in graphs:
        res.append(mapping_filter[filter_type](graph, param))
    return res


def filter_by_threshold(graph: nx.Graph, threshold: float) -> nx.Graph:
    edges = graph.edges
    amount_of_edges = len([edge for edge in edges if edges[edge]['weight'] < threshold])
    return filter_by_amount(graph, amount_of_edges)


def filter_by_pmfg(graph: nx.Graph, param: int = 0) -> nx.Graph:
    amount_of_nodes = len(graph.nodes)
    amount_of_edges = 3 * (amount_of_nodes - 2)
    sorted_edges = sort_graph_edges(graph)
    sorted_edges.reverse()
    pmfg = nx.Graph()
    nodes_with_attr = [(node, {'label': node['label']}) for node in list(graph.nodes())]
    pmfg.add_nodes_from(nodes_with_attr)

    for edge in sorted_edges:
        pmfg.add_edge(edge['source'], edge['dest'])
        if not nx.check_planarity(pmfg):
            pmfg.remove_edge(edge['source'], edge['dest'])
        if len(pmfg.edges()) == amount_of_edges:
            return pmfg


def filter_by_dens(graph: nx.Graph, density: float) -> nx.Graph:
    amount_of_nodes = len(graph.nodes())
    amount_of_edges = (amount_of_nodes * (amount_of_nodes - 1)) / 2
    return filter_by_amount(graph, int(amount_of_edges * density))


def filter_by_amount(graph: nx.Graph, amount_edges: int) -> nx.Graph:
    sorted_edges = sort_graph_edges(graph)
    norm_g = deepcopy(graph)
    norm_g.remove_edges_from([sorted_edges[:-amount_edges]])
    return norm_g


def sort_graph_edges(graph: nx.Graph) -> nx.edges:
    edges = sorted(graph.edges(data=True), key=lambda t: t[2].get('weight', 1))
    return edges





