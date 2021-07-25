import numpy as np
from sklearn.model_selection import LeaveOneOut
import copy
from conf_pack.configuration import tune_parameters
from pre_process import build_graphs_from_corr, load_scans, create_graphs_features_df
from feature_extraction import main_global_features, features_by_type
from train import train_model, predict_by_criterions, info_gain_all_features
import nilearn
from utils import *
from conf_pack.opts import parser
from collections import defaultdict

from visualization import build_features_for_scatters, scatter_plot, hist_class

k = 100


def main():

    performances = defaultdict(list)
    labels = get_y_true()
    min_feat = default_params.getint('min_features')
    max_feat = default_params.getint('max_features')
    filter_type = default_params.get('filter')
    is_globals = default_params.get('features_type') == 'globals'

    if not is_globals:
        graphs = get_graphs(get_corr_lst(), tune_parameters[filter_type])

    for thresh in tune_parameters[filter_type]:

        if is_globals:
            features = load_graphs_features(filter_type, thresh)
            print(thresh)

        for feat_num in range(min_feat, max_feat):

            if not is_globals:
                features = features_by_type(default_params.get('features_type'), graphs[thresh], feat_num)

            acc, _, _ = train_model(features, labels, feat_num)
            performances[(thresh, feat_num)].append(acc)
    save_results(performances)

    config_update({filter_type: tune_parameters[filter_type]})


def fetch_data_example():
    data = nilearn.datasets.fetch_adhd(n_subjects=40, data_dir='C:/Users/orsym/Documents/Data/ADHD')
    return data


def hyper_parameter(hyper_parameters: Dict):
    # Todo: Another table like count table only for features. Each feature is a column.

    loo = LeaveOneOut()
    performances, counts_table, features_table, y, avg_acc, corr_lst, filter_type = initialize_hyper_parameters()
    is_globals = default_params.get('features_type') == 'globals'
    if not is_globals:
        graphs = get_graphs(get_corr_lst(), tune_parameters[filter_type])

    for train_idx, test_idx in loo.split(corr_lst):

        best_thresh, best_acc, best_num, best_model, feat_names_best = 0, 0, 0, None, None

        for criteria_thresh in hyper_parameters['threshold']:

            if is_globals:
                df = load_graphs_features(filter_type, criteria_thresh)

            # info_gain_all_features(df, y, threshold=criteria_thresh)

            for num_features in hyper_parameters['num_features']:

                if not is_globals:
                    df = features_by_type(default_params.get('features_type'), graphs[criteria_thresh], num_features)
                X_train, X_test = df.iloc[train_idx], df.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

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
    create_stability_df(counts_table_refactored)
    feat_table_refactored = dict_to_df(features_table, 'feature', 'num_counts', 'feat_count_table.csv')
    feat_table_refactored.sort_values(by='num_counts', inplace=True)

    for i in range(0, 5):
        plot_hyper_parameters(feat_table_refactored, filter_type, hyper_parameters, i)

    with open(os.path.join(get_results_path(), 'Results.txt'), 'a') as f:
        f.write(f'The accuracy of this experiment is {avg_acc}\n')

        # pd.DataFrame(performances).to_csv(os.path.join(get_results_path(), 'hyper_parameters.csv'), index=False)
    config_to_save = copy.deepcopy(hyper_parameters)

    config_update(config_to_save)

    return performances


def config_update(config_to_save):
    config_to_save.update({'filtering criteria': [default_params.get('filter')], 'class predict': \
        [default_params.get('class_name')], 'features_type': [default_params.get('features_type')]})
    save_config(config_to_save)


def plot_hyper_parameters(feat_table_refactored, filter_type, hyper_parameters, i):
    feat_name_to_plot = feat_table_refactored.iloc[i]['feature']
    scatter_plot(build_features_for_scatters(filter_type, hyper_parameters['threshold'],
                                             feat_name_to_plot, get_y_true_regression()), feat_name_to_plot)
    hist_class(build_features_for_scatters(filter_type, hyper_parameters['threshold'], feat_name_to_plot,
                                           get_y_true()), feat_name_to_plot)


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

def get_corr_lst():
    data_path = os.path.join(get_data_path(), 'nifti')
    names = [os.path.join(data_path, name) for name in os.listdir(data_path)]
    corr_lst = load_scans(names, get_data_path())
    return corr_lst


def get_graphs(corr_lst: List[np.ndarray], params: List[float]) -> Dict:
    graphs_by_param = {}
    for param in params:
        graphs_by_param[param] = build_graphs_from_corr(default_params.get('filter'), corr_lst, param)
    return graphs_by_param

if __name__ == '__main__':
    main()
    # data = fetch_data_example()
    # graph_pre_process()
    start = 0.5
    stop = 0.51
    # config = {'threshold': [0.43, 0.44], 'num_features': [6]}
    # config = {'threshold': list(set(list(np.arange(start, stop, step=0.01)) + list(np.arange(0.42, 0.43, step=0.01)))),
    #           'num_features': list(range(1, 2))}

    # hyper_parameter(config)
