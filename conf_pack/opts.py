import argparse


parser = argparse.ArgumentParser(description='Balance prediction experiment')
parser.add_argument('--min_features', help='Minimum number of features to take in feature selection',
                    default=1, type=int)
parser.add_argument('--max_features', help='Maximum, number of features to take in feature selection',
                    default=10, type=int)
parser.add_argument('--exp_type', help='Experiment type. Can be T1, T2 or Delta', default='T1', type=str)
parser.add_argument('--min_threshold', help='Minimum threshold for remove edges in the edge criteria',
                    default=0.40, type=float)
parser.add_argument('--max_threshold', help='Maximum threshold for remove edges in the edge criteria',
                    default=0.41, type=float)
parser.add_argument('--criteria', help='Criteria to use for the edge filtering', default='pmfg', type=str)
parser.add_argument('--task', default='derive_T1', type=str)
parser.add_argument('--f_type', default='globals', type=str)
parser.add_argument('--step', default='0.01', type=float)