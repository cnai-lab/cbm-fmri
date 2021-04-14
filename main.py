from pre_process import filter_edges, load_scans
import os
from paths import *
k = 100
thresholds = [thresh / k for thresh in range(40, 70)]


def example():
    thresh = 0.4
    graphs = load_scans(os.listdir(SCANS_DIR))

