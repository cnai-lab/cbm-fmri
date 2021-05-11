from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from typing import Union, Tuple
import pandas as pd
from sklearn.metrics import accuracy_score
from typing import List
import numpy as np
from sklearn import preprocessing
from torch import nn

from Deep.model import Net
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from multiprocessing import Process, Pool
from utils import write_selected_features, load_graphs_features


def train_model(df: pd.DataFrame, y: np.ndarray, num_features: int) -> Tuple[float, RandomForestClassifier, List[str]]:
    loo = LeaveOneOut()
    df = df.fillna(0)
    df = normalize_features(df)
    avg_acc = 0
    args_lst = []

    for train_idx, test_idx in loo.split(df):
        X_train, X_test = df.iloc[train_idx], df.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        args_lst.append((X_train, y_train, X_test, y_test, num_features))
    with Pool(5) as p:
        res = p.starmap(train_model_iteration, args_lst)
    avg_acc = sum(res)

    model = load_model('rf')
    feat_names, feat_values = select_features(df, y, num_features)
    model.fit(df[feat_names], y)

    avg_acc /= len(y)
    print(avg_acc)
    return avg_acc, model, feat_names


def train_model_iteration(X_train: pd.DataFrame, y_train: np.ndarray,
                          X_test: pd.DataFrame, y_test: np.ndarray, num_features: int) -> float:

    model = load_model('rf')

    feat_names, feat_values = select_features(X_train, y_train, num_features)
    # write_selected_features(feat_names, feat_values)
    X_train, X_test = X_train[feat_names], X_test[feat_names]
    model.fit(X_train, y_train)
    return accuracy_score(model.predict(X_test), y_test)




def predict_by_criterions(model, col_names: List[str], filter_type: str, thresh: float, idx: np.ndarray,
                          y: np.ndarray) -> float:
    df = load_graphs_features(filter_type, thresh)
    df = df[col_names]
    df = df.fillna(0)
    df = normalize_features(df)
    df_relevant_features = df.iloc[idx]
    y_relevant = y[idx]
    acc = accuracy_score(model.predict(df_relevant_features), y_relevant)
    return acc


def load_model(model_type: str) -> Union[nn.Module, RandomForestClassifier, None]:
    if model_type == 'deep':
        model = Net()
    elif model_type == 'rf':
        model = RandomForestClassifier(n_estimators=300)
    else:
        raise ValueError('Invalid model_type as input')
    return model


def select_features(x_train: pd.DataFrame, y_true: np.ndarray, num_features: int) -> Tuple[List[str], List[float]]:
    def inf_gain(X, y):
        return mutual_info_classif(X, y, random_state=42)
    selector = SelectKBest(inf_gain, k=num_features).fit(x_train, y_true)
    mask = selector.get_support()
    values = mutual_info_classif(x_train, y_true, random_state=42)[mask]
    feature_names = x_train.columns[mask]
    return feature_names, values


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    min_max = preprocessing.MinMaxScaler()
    df[df.columns] = min_max.fit_transform(df)
    return df
