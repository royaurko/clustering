import cluster
from sklearn.datasets import load_iris
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--metric', default='avg', help='Can be one of {avg, max, min}')
    parser.add_argument('-o', '--out_file', default='result.pkl', help='Pickle file to store the result')
    parser.add_argument('-n', '--num_workers', type=int, default=1, help='Number of workers')
    args = parser.parse_args()
    d = load_iris()
    data = d['data']
    target = d['target']
    cluster.main(data, target, args.metric, args.out_file, args.num_workers)
