import pandas as pd
from sklearn.model_selection import LeaveOneOut

from pre_process import build_graphs_from_corr, load_scans, create_graphs_features_df
from feature_extraction import main_global_features
from train import train_model, predict_by_criterions
import nilearn
from utils import *
from collections import defaultdict
k = 100


def main():

    performances = defaultdict(list)
    data_path = os.path.join(get_data_path(), 'nifti')
    names = [os.path.join(data_path, name) for name in os.listdir(data_path)]

    corr_lst = load_scans(names)
    filter_type = default_params.get('filter')

    for thresh in tune_parameters[filter_type]:
        graphs = build_graphs_from_corr(corr_lst=corr_lst, filter_type=filter_type, param=thresh)
        features = main_global_features(graphs)
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
    performances, counts_table = defaultdict(list), defaultdict(int)
    y = get_y_true()
    avg_acc = 0

    data_path = os.path.join(get_data_path(), 'nifti')
    names = [os.path.join(data_path, name) for name in os.listdir(data_path)]
    corr_lst = load_scans(names)

    loo = LeaveOneOut()
    filter_type = default_params.get('filter')

    for train_idx, test_idx in loo.split(corr_lst):

        best_thresh, best_acc, best_num, best_model = 0, 0, 0, None

        for criteria_thresh in hyper_parameters['threshold']:

            df = load_graphs_features(filter_type, criteria_thresh)
            X_train, X_test = df.iloc[train_idx], df.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            for num_features in hyper_parameters['num_features']:

                acc, model, feat_names = train_model(X_train, y_train, num_features)

                if acc > best_acc:
                    best_thresh, best_num, best_model = criteria_thresh, num_features, model

        avg_acc += predict_by_criterions(model=best_model, filter_type=filter_type, thresh=best_thresh, idx=test_idx,
                                         y=y, col_names=feat_names)

        counts_table[(best_thresh, best_num)] = counts_table[(best_thresh, best_num)] + 1
    avg_acc /= len(y)
    with open(os.path.join(get_results_path(), 'Results.txt')) as f:
        f.write(f'The accuracy of this experiment is {avg_acc}\n')
    count_table_refactored = defaultdict(list)
    for key, val in counts_table.items():
        count_table_refactored['params'].append(key)
        count_table_refactored['num_counts'].append(val)
    pd.DataFrame(count_table_refactored).to_csv(os.path.join(get_results_path()), 'count_table.csv')

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
    corr_lst = load_scans(names)

    create_graphs_features_df(corr_lst=corr_lst, filter_type='threshold', thresholds=np.arange(start=0.4, stop=0.7,
                                                                                   step=0.01))


if __name__ == '__main__':
    # data = fetch_data_example()
    hyper_parameter({'threshold': [0.43], 'num_features': [1]})
    # graph_pre_process()