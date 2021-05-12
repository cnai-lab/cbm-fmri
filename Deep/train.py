import torch
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from Deep.dataloader import GraphsDataset
from Deep.model import load_model
from torch import nn
from torch.utils.data import DataLoader
from typing import NoReturn
from Deep.model_utils import load_criteria, load_optimizer
from conf_pack.configuration import default_params
from utils import get_data_path
# Train -> Dataloader -> Pre-process -> utils -> configuration
# Train -> utils

def main_train():
    dataset = GraphsDataset(root=get_data_path())
    model = load_model(num_feat=218, num_classes=2)
    dl = DataLoader(dataset, batch_size=default_params.getint('batch_size'))
    train_model(model, dl)


def train_model(model: nn.Module, dl: DataLoader) -> NoReturn:
    criteria = load_criteria()
    optimizer = load_optimizer(model)
    with torch.enable_grad:
        for graph, label, filename in dl:
            optimizer.zero_grad()
            output = model(graph)
            print(output.shape)
            loss = criteria(output, label)
            output = F.softmax(output)
            _, preds = torch.argmax(output, dim=1)
            print(loss.item())
            print(accuracy_score(y_true=label, y_pred=preds))
            optimizer.step()


if __name__ == '__main__':
    main_train()