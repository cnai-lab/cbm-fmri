import pandas as pd
from sklearn.model_selection import LeaveOneOut

from conf_pack.configuration import tune_parameters
from pre_process import build_graphs_from_corr, load_scans, create_graphs_features_df
from feature_extraction import main_global_features
from train import train_model, predict_by_criterions, info_gain_all_features
import nilearn
from utils import *

from collections import defaultdict

from visualization import build_features_for_scatters, scatter_plot

k = 100


def main():

    performances = defaultdict(list)
    data_path = os.path.join(get_data_path(), 'nifti')
    names = [os.path.join(data_path, name) for name in get_names()]

    # corr_lst = load_scans(names)
    filter_type = default_params.get('filter')

    for thresh in tune_parameters[filter_type]:
        # graphs = build_graphs_from_corr(corr_lst=corr_lst, filter_type=filter_type, param=thresh)
        # features = main_global_features(graphs)
        features = load_graphs_features(filter_type, thresh)
        labels = get_y_true()
        print(thresh)
        for feat_num in range(default_params.getint('min_features'), default_params.getint('max_features')):
            acc, _, _ = train_model(features, labels, feat_num)
            performances[(thresh, feat_num)].append(acc)

    save_results(performances)

def fetch_data_example():
    data = nilearn.datasets.fetch_adhd(n_subjects=40, data_dir='C:/Users/orsym/Documents/Data/ADHD')
    return data


def hyper_parameter(hyper_parameters: dict):
    # Todo: Another table like count table only for features. Each feature is a column.
    performances, counts_table, features_table = defaultdict(list), defaultdict(int), defaultdict(int)
    y = get_y_true()
    avg_acc = 0

    data_path = os.path.join(get_data_path(), 'nifti')
    names = [os.path.join(data_path, name) for name in get_names()]
    corr_lst = load_scans(names, get_data_path())

    loo = LeaveOneOut()
    filter_type = default_params.get('filter')

    for train_idx, test_idx in loo.split(corr_lst):

        best_thresh, best_acc, best_num, best_model, feat_names_best = 0, 0, 0, None, None

        for criteria_thresh in hyper_parameters['threshold']:

            df = load_graphs_features(filter_type, criteria_thresh)
            info_gain_all_features(df, y, threshold=criteria_thresh)
            X_train, X_test = df.iloc[train_idx], df.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            for num_features in hyper_parameters['num_features']:

                acc, model, feat_names = train_model(X_train, y_train, num_features)

                if acc > best_acc:
                    best_acc = acc
                    feat_names_best = feat_names
                    best_thresh, best_num, best_model = criteria_thresh, num_features, model

        avg_acc += predict_by_criterions(model=best_model, filter_type=filter_type, thresh=best_thresh, idx=test_idx,
                                         y=y, col_names=feat_names_best)

        counts_table[(best_thresh, best_num)] = counts_table[(best_thresh, best_num)] + 1

        for feat in feat_names_best:
            features_table[feat] = features_table[feat] + 1

    avg_acc /= len(y)

    counts_table_refactored = dict_to_df(counts_table, 'params', 'num_counts', 'count_table.csv')
    feat_table_refactored = dict_to_df(features_table, 'feature', 'num_counts', 'feat_count_table.csv')
    feat_table_refactored.sort_values(by='num_counts', inplace=True)

    for i in range(0, 1):
        feat_name_to_plot = feat_table_refactored.iloc[i]['feature']
        scatter_plot(build_features_for_scatters(filter_type, hyper_parameters['threshold'],
                                                 feat_name_to_plot, y), feat_name_to_plot)

    with open(os.path.join(get_results_path(), 'Results.txt'), 'a') as f:
        f.write(f'The accuracy of this experiment is {avg_acc}\n')

        # pd.DataFrame(performances).to_csv(os.path.join(get_results_path(), 'hyper_parameters.csv'), index=False)
    return performances



def example():

    thresh = 0.4
    corr_lst = load_scans([os.path.join(SCANS_DIR_BEFORE, name) for name in os.listdir(SCANS_DIR_BEFORE)])
    graphs = build_graphs_from_corr(corr_lst=corr_lst, filter_type='threshold', param=thresh)
    main_global_features(graphs)

    print()


def graph_pre_process():
    data_path = os.path.join(get_data_path(), 'nifti')
    names = [os.path.join(data_path, name) for name in os.listdir(data_path)]
    corr_lst = load_scans(names, get_data_path())

    create_graphs_features_df(corr_lst=corr_lst, filter_type='threshold', thresholds=np.arange(start=0.4, stop=0.7,
                                                                                   step=0.01))


if __name__ == '__main__':
    # main()
    # data = fetch_data_example()
    hyper_parameter({'threshold': [0.43], 'num_features': [2]})
    # graph_pre_process()