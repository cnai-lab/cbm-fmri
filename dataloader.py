from torch_geometric.data import Data, Dataset, DataLoader, InMemoryDataset
import networkx as nx
from typing import List, Tuple
from torch_geometric.utils import from_networkx
import pandas as pd
import os
from pre_process import load_scans, build_graphs_from_corr


class GraphsDataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(GraphsDataset, self).__init__(root, transform, pre_transform)
        self.root = root
        self.csv = pd.read_csv(os.path.join(root, 'Data.csv'))
        self.filenames = self.csv['Subject']
        self.labels = self.csv['Class']




    def len(self):
        return len(self.csv)

    def get(self, idx: int) -> Tuple[Data, int]:
        full_path = os.path.join(self.root, f'{self.filenames[idx]}.nii')
        data_lst = load_scans([full_path], 'both')
        activations = data_lst[0][0].swapaxes(0, 1)
        correlation = data_lst[1][0]
        graph = build_graphs_from_corr('density', [correlation], 0.01)[0]
        nx.set_node_attributes(graph, dict(zip(range(len(activations)), activations)), 'activations')
        # Todo: understand why the label return is int type 64 and not int.
        return from_networkx(graph), int(self.labels[idx])


def nx_lst_to_dl(graphs: List[nx.Graph]) -> DataLoader:
    lst_torch_graphs = []
    for graph in graphs:
        torch_graph = from_networkx(graph)
        lst_torch_graphs.append(torch_graph)
    dl = DataLoader(lst_torch_graphs)
    return dl


if __name__ == '__main__':
    dataset = GraphsDataset(root=path)
    dl = DataLoader(dataset)
    for graph, label in dl:
        print(label)
