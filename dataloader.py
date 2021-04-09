from torch_geometric.data import Data, Dataset, DataLoader
import networkx as nx
from typing import List
from torch_geometric.utils import from_networkx
import pandas as pd
import os
from pre_process import load_scans

class GraphsDataset(Dataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(GraphsDataset, self).__init__(root, transform, pre_transform)
        self.root = root
        self.csv = pd.read_csv(os.path.join(root, 'data.csv'))
        self.filenames = self.csv['names']
        self.labels = self.csv['labels']


    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        return len(self.csv)

    def get(self, idx):
        full_path = os.path.join(self.root, self.filenames[idx])
        load_scans(full_path)

    @property
    def raw_file_names(self):
        pass


def nx_lst_to_dl(graphs: List[nx.Graph]) -> DataLoader:
    lst_torch_graphs = []
    for graph in graphs:
        torch_graph = from_networkx(graph)
        lst_torch_graphs.append(torch_graph)
    dl = DataLoader(lst_torch_graphs)
    return dl


