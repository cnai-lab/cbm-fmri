from pre_process import build_graphs_from_corr, load_scans
import os
from paths import *
k = 100



def example():
    thresh = 0.4
    corr_lst = load_scans([os.path.join(SCANS_DIR_BEFORE, name) for name in os.listdir(SCANS_DIR_BEFORE)])
    graphs = build_graphs_from_corr(corr_lst=corr_lst, filter_type='threshold', param=thresh)

    print()


if __name__ == '__main__':
    example()