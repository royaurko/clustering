import cluster
import argparse
import numpy as np
import csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--metric', default='avg', help='Can be one of {avg, max, min}')
    parser.add_argument('-o', '--out_file', default='result.pkl', help='Pickle file to store the result')
    parser.add_argument('-n', '--num_workers', type=int, default=1, help='Number of workers')
    args = parser.parse_args()
    path = 'spambase/spambase.data'
    reader = csv.reader(open(path), delimiter=',')
    data = []
    target = []
    for row in reader:
        data.append(row[:-1])
        target.append(row[-1])
    data = np.array(data, dtype=float)
    labels = set(target)
    label_to_idx = {v: i for i, v in enumerate(labels)}
    target = np.array([label_to_idx[i] for i in target], dtype=int)
    cluster.main(data, target, args.metric, args.out_file, args.num_workers)
