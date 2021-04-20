import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

