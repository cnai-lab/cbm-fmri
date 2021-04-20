# from sklearn.ensemble import RandomForestClassifier
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.data.data import Data
import torch.nn.functional as F
#


class Net(nn.Module):
    def __init__(self, num_features, num_classes):
        super(Net).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data: Data):
        x, edge_idx = data.edge_index
        x = self.conv1(x, edge_idx)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_idx)
        return F.log_softmax(x, dim=1)
