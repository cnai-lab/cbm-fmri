from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Union
import pandas as pd
from sklearn.metrics import accuracy_score
from typing import List
import numpy as np
from sklearn import preprocessing
from torch import nn
from model import Net
from sklearn.feature_selection import mutual_info_classif, SelectKBest

from utils import write_selected_features


def train_model(df: pd.DataFrame, y: np.ndarray, num_features: int) -> float:
    model = load_model('rf')
    loo = LeaveOneOut()
    df = df.fillna(0)
    df = normalize_features(df)
    avg_acc = 0

    for train_idx, test_idx in loo.split(df):

        X_train, X_test = df.iloc[train_idx], df.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        feat_names = select_features(X_train, y_train, num_features)
        write_selected_features(feat_names)
        X_train, X_test = X_train[feat_names], X_test[feat_names]
        model.fit(X_train, y_train)
        avg_acc += accuracy_score(model.predict(X_test), y_test)

    avg_acc /= len(y)
    print(avg_acc)
    return avg_acc


def load_model(model_type: str) -> Union[nn.Module, RandomForestClassifier, None]:
    if model_type == 'deep':
        model = Net()
    elif model_type == 'rf':
        model = RandomForestClassifier(n_estimators=300)
    else:
        raise ValueError('Invalid model_type as input')
    return model


def select_features(x_train: pd.DataFrame, y_true: np.ndarray, num_features: int) -> List[str]:
    selector = SelectKBest(mutual_info_classif, k=num_features).fit(x_train, y_true)
    mask = selector.get_support()
    feature_names = x_train.columns[mask]
    return feature_names


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    min_max = preprocessing.MinMaxScaler()
    df[df.columns] = min_max.fit_transform(df)
    return df
