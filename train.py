import os
from collections import defaultdict

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

from conf_pack.configuration import default_params
from feature_extraction import features_by_type
from utils import write_selected_features, load_graphs_features, get_results_path


def train_model(df: pd.DataFrame, y: np.ndarray, num_features: int) -> \
        Tuple[float, RandomForestClassifier, List[str]]:
    loo = LeaveOneOut()
    df = df.fillna(0)
    df = normalize_features(df)
    avg_acc = 0
    args_lst = []
    res = []

    for train_idx, test_idx in loo.split(df):
        X_train, X_test = df.iloc[train_idx], df.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        args_lst.append((X_train, y_train, X_test, y_test, num_features))
        res.append(train_model_iteration(X_train, y_train, X_test, y_test, num_features))
        # with Pool(1) as p:
    #     res = p.starmap(train_model_iteration, args_lst)
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
    write_selected_features(feat_names, feat_values)
    X_train, X_test = X_train[feat_names], X_test[feat_names]
    model.fit(X_train, y_train)
    return accuracy_score(model.predict(X_test), y_test)


def predict_by_criterions(**kwargs) -> float:

    col_names = kwargs['col_names']
    df = kwargs['df']
    idx = kwargs['idx']
    y = kwargs['y']
    model = kwargs['model']

    df = df[col_names]
    df = df.fillna(0)
    df = normalize_features(df)
    df_relevant_features = df.iloc[idx]
    y_relevant = y[idx]

    acc = accuracy_score(model.predict(df_relevant_features), y_relevant)
    return acc


def predict_by_proba(model, X_test, thresh: float = 0.6):
    rf = RandomForestClassifier()
    rf.predict_proba(X_test)
    prob = model.predict_proba(X_test)[0]
    if prob > thresh:
        return 1
    elif prob < (1 - thresh):
        return 0
    else:
        return -1

        # Between 1-thresh to thresh

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
        return mutual_info_classif(X, y)
    if default_params.get('features_type') == 'globals':

        selector = SelectKBest(inf_gain, k=num_features).fit(x_train, y_true)
        mask = selector.get_support()
        values = mutual_info_classif(x_train, y_true)[mask]
        feature_names = x_train.columns[mask]
        return feature_names, values
    else:
        return list(x_train.columns), list(np.ones(len(x_train.columns)))

def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    min_max = preprocessing.MinMaxScaler()
    df[df.columns] = min_max.fit_transform(df)
    return df


def info_gain_all_features(df: pd.DataFrame, y_true: np.ndarray, threshold: float):
    df_res = defaultdict(list)
    values = mutual_info_classif(X=df.fillna(0), y=y_true)
    for col, val in zip(df.columns, values):
        df_res[col].append(val)
    df_res['threshold'].append(threshold)
    full_path = os.path.join(get_results_path(), 'all_features.csv')
    if os.path.exists(full_path):
        pd.DataFrame(df_res, index=df_res['threshold']).to_csv(full_path, header=False, mode='a')
    else:
        pd.DataFrame(df_res, index=df_res['threshold']).to_csv(full_path)


if __name__ == '__main__':
    df = pd.DataFrame({'values': [1.53, 1.54, 1.78, 2.4],
                       'target': [2, 2, 1, 2]})
    mutual_info_classif(df[['values']], df['target'])