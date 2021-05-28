import os
from typing import Dict, List, NoReturn, DefaultDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
import seaborn as sns

from utils import get_save_path, load_graphs_features, get_results_path


def box_plot(paths, col_name, task, y_col, criteria, title):
    plt.clf()
    df_res = pd.DataFrame()
    for path in paths:
        df = pd.read_csv(path)
        path_parts = path.split("/")

        df['experiment type'] = path[path.rfind('_') + 1:path.rfind('.')]
        df_res = pd.concat([df_res, df])

    fig = sns.boxplot(y=y_col, x='experiment type', data=df_res, showmeans=True)
    fig.set_title(title)
    # fig.set_xticklabels(fig.get_xticklabels(), fontsize=9)
    fig.set_xlabel('Experiment Type', fontsize=16)
    fig.set_ylabel(col_name, fontsize=16)
    fig.set_xticklabels(fig.get_xticklabels(), rotation=40, ha="right", fontsize=14)
    plt.tight_layout()
    plt.figure(figsize=(4, 2))
    fig.figure.savefig(f'box_plot_{criteria}_{task}_{y_col}.png')
    plt.close()


def build_features_for_scatters(filter_type: str, thresh_lst: List[float], col_name: str, y: np.ndarray) -> DefaultDict:
    res = defaultdict(dict)
    for thresh in thresh_lst:
        df = load_graphs_features(filter_type, thresh)
        df_relevant = df[col_name].values
        for i in range(len(df)):
            res[i]['values'] = res[i].get('values', []) + [df_relevant[i]]
            res[i]['target'] = res[i].get('target', []) + [y[i]]
    return res


def scatter_plot(features: Dict, feature_name: str) -> NoReturn:
    colors = np.arange(0, 105, 5)
    colors_dict = {key: val for key, val in zip(features.keys(), colors)}

    for key, item in features.items():
        plt.scatter(x=features[key]['values'], y=features[key]['target'],
                    c=colors_dict[key] * len(features[key]['target']),  cmap='viridis')

    plt.savefig(os.path.join(get_results_path(), f'scatter_graph_{feature_name}.png'))