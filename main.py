import pandas as pd
from sklearn.model_selection import LeaveOneOut

from pre_process import build_graphs_from_corr, load_scans
from feature_extraction import main_global_features
from train import train_model
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
            acc = train_model(features, labels, feat_num)
            performances[(thresh, feat_num)].append(acc)

    save_results(performances)

def fetch_data_example():
    data = nilearn.datasets.fetch_adhd(n_subjects=40, data_dir='C:/Users/orsym/Documents/Data/ADHD')
    return data


def hyper_parameter(hyper_parameters: dict):
    performances = defaultdict(list)
    y = get_y_true()
    data_path = os.path.join(get_data_path(), 'nifti')
    names = [os.path.join(data_path, name) for name in os.listdir(data_path)]

    corr_lst = load_scans(names)
    loo = LeaveOneOut()
    filter_type = default_params.get('filter')
    for train_idx, test_idx in loo.split(corr_lst):

        for criteria_thresh in hyper_parameters['threshold']:
            train_corr_lst = [item for idx, item in enumerate(corr_lst) if idx != test_idx[0]]
            graphs = build_graphs_from_corr(corr_lst=train_corr_lst, filter_type=filter_type, param=criteria_thresh)

            df = main_global_features(graphs)

            for num_features in hyper_parameters['num_features']:

                X_train, X_test = df.iloc[train_idx], df.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                performances['threshold'].append(criteria_thresh)
                performances['num_features'].append(num_features)
                performances['model_accuracy'].append(train_model(X_train, y_train, num_features))
                pd.DataFrame(performances).to_csv(os.path.join(get_results_path(), 'hyper_parameters.csv'), index=False)
        return performances



def example():

    thresh = 0.4
    corr_lst = load_scans([os.path.join(SCANS_DIR_BEFORE, name) for name in os.listdir(SCANS_DIR_BEFORE)])
    graphs = build_graphs_from_corr(corr_lst=corr_lst, filter_type='threshold', param=thresh)
    main_global_features(graphs)

    print()


if __name__ == '__main__':
    # data = fetch_data_example()
    hyper_parameter({'threshold': [0.43], 'num_features': [1]})