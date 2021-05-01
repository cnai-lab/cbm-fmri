import networkx as nx
import nilearn
from conf_pack.paths import DATA_PATH
# def build_graphs()
import os
from pre_process import load_scans


if __name__ == '__main__':
    nilearn.datasets.fetch_abide_pcp()
