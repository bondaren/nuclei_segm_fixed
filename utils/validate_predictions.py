import csv

import numpy as np
import h5py
import os
# Add new metrics if needed
from utils.voi import voi

from utils.rand import adapted_rand


def count_cells(seg, gt):
    seg_count = len(np.unique(seg)) - 1
    gt_count = len(np.unique(gt)) - 1
    return (seg_count, gt_count)


def filter_small_objects(labels, min_size=500):
    ids, counts = np.unique(labels, return_counts=True)
    # divide labels between 'small' and 'large' regions
    small_objects = []
    for i, c in zip(ids, counts):
        if c < min_size:
            small_objects.append(i)
    print(f'Small objects: {small_objects}')
    # zero out small regions
    for i in small_objects:
        labels[labels == i] = 0

    return labels


def write_csv(output_path, results):
    assert len(results) > 0
    keys = results[0].keys()
    print(f'Saving results to {output_path}...')
    with open(output_path, "w") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)


metrics = {
    "voi": voi,
    "arand_error": adapted_rand,
    "count": count_cells
}

if __name__ == "__main__":
    gt_files = [
        '/g/kreshuk/wolny/Datasets/Vladyslav/GT/test/GT_Ab1_test.h5',
        '/g/kreshuk/wolny/Datasets/Vladyslav/GT/test/GT_Ab1_test.h5',
        '/g/kreshuk/wolny/Datasets/Vladyslav/GT/test/GT_Ab1_test.h5',
        '/g/kreshuk/wolny/Datasets/Vladyslav/GT/test/GT_Ab2_test.h5',
        '/g/kreshuk/wolny/Datasets/Vladyslav/GT/test/GT_Ab2_test.h5',
        '/g/kreshuk/wolny/Datasets/Vladyslav/GT/test/GT_Ab2_test.h5',
    ]
    seg_files = [
        '/g/kreshuk/wolny/Datasets/Vladyslav/GT/test/GT_Ab1_test_predictions_mc.h5',
        '/g/kreshuk/wolny/Datasets/Vladyslav/GT/test/GT_Ab1_test_predictions_mws.h5',
        '/g/kreshuk/wolny/Datasets/Vladyslav/GT/test/GT_Ab1_test_predictions_threshold.h5',
        '/g/kreshuk/wolny/Datasets/Vladyslav/GT/test/GT_Ab2_test_predictions_mc.h5',
        '/g/kreshuk/wolny/Datasets/Vladyslav/GT/test/GT_Ab2_test_predictions_mws.h5',
        '/g/kreshuk/wolny/Datasets/Vladyslav/GT/test/GT_Ab2_test_predictions_threshold.h5'
    ]

    gt_seg_cache = {}

    results = []
    for seg_file, gt_file in zip(seg_files, gt_files):
        print(f'Loading GT from', gt_file)
        if not gt_file in gt_seg_cache:
            print(f'Saving {gt_file} in the cache')
            with h5py.File(gt_file, 'r') as f:
                gt = f['label'][...]
                gt = filter_small_objects(gt)
                gt_seg_cache[gt_file] = gt

        gt = gt_seg_cache[gt_file]

        print(f'Loading SEG from', seg_file)
        with h5py.File(seg_file, 'r') as f:
            seg = f['segmentation'][...]
            seg = filter_small_objects(seg)

        print(f'Computing metrics')
        scores = {
            'seg_file': os.path.split(seg_file)[1],
            'gt_file': os.path.split(gt_file)[1]
        }

        for key, metric in metrics.items():
            result = metric(seg, gt)
            scores[key] = result
            print(key, ": ", result)
        results.append(scores)

    write_csv('segmentation_metrics.csv', results)
