import torch

from Deep.dataloader import GraphsDataset
from Deep.model import load_model
from torch import nn
from torch.utils.data import DataLoader
from typing import NoReturn

from conf_pack.configuration import default_params
from utils import get_data_path


def main_train():
    dataset = GraphsDataset(root=get_data_path())
    model = load_model(num_feat=218, num_classes=2)
    dl = DataLoader(dataset, batch_size=default_params.getint('batch_size'))


def train_model(model: nn.Module, dl: DataLoader) -> NoReturn:
    with torch.enable_grad:
        for graph, label, filename in dl:
            model(graph)
            print(f'label of {filename} is {label}')
