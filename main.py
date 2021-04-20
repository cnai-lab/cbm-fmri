from pre_process import build_graphs_from_corr, load_scans
from feature_extraction import main_global_features
from train import train_model
from utils import *
from collections import defaultdict
k = 100


def main():

    performances = defaultdict(list)
    names = [os.path.join(SCANS_DIR_BEFORE, name) for name in os.listdir(SCANS_DIR_BEFORE)]
    corr_lst = load_scans(names)
    filter_type = default_params.get('filter')

    for thresh in tune_parameters[filter_type]:
        graphs = build_graphs_from_corr(corr_lst=corr_lst, filter_type=filter_type, param=thresh)
        features = main_global_features(graphs)
        labels = get_y_true()

        for feat_num in range(default_params.getint('min_features'), default_params.getint('max_features')):
            acc = train_model(features, labels, feat_num)
            performances[(thresh, feat_num)].append(acc)

    save_results(performances)

def example():

    thresh = 0.4
    corr_lst = load_scans([os.path.join(SCANS_DIR_BEFORE, name) for name in os.listdir(SCANS_DIR_BEFORE)])
    graphs = build_graphs_from_corr(corr_lst=corr_lst, filter_type='threshold', param=thresh)
    main_global_features(graphs)

    print()


if __name__ == '__main__':
    example()