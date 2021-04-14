from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Union
import pandas as pd
from sklearn.metrics import accuracy_score,auc
import numpy as np
from torch import nn
from model import Net


def train_model(df: pd.DataFrame, y: np.ndarray) -> Dict:
    model = load_model('rf')
    loo = LeaveOneOut()
    measurements = {}
    avg_acc, avg_auc = 0, 0
    for train_idx, test_idx in loo.split(df):
        X_train, X_test = df[train_idx], df[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        avg_acc += accuracy_score(model.predict(X_test), y_test)
        avg_auc += auc(X_test, y_test)
    avg_acc /= len(y)
    avg_auc /= len(y)
    measurements['acc'] = avg_acc
    measurements['auc'] = avg_auc
    return measurements


def load_model(model_type: str) -> Union[nn.Module, RandomForestClassifier, None]:
    if model_type == 'deep':
        model = Net()
    elif model_type == 'rf':
        model = RandomForestClassifier()
    else:
        raise ValueError('Invalid model_type as input')
    return model

