import argparse
import logging
import sys
import os

import h5py
import matplotlib.pyplot as plt

plt.ioff()
plt.switch_backend('agg')

logger = logging.getLogger('Metrics')
logger.setLevel(logging.INFO)
# Logging to console
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

import numpy as np
from skimage.metrics import contingency_table


def precision(tp, fp, fn):
    return tp / (tp + fp) if tp > 0 else 0


def recall(tp, fp, fn):
    return tp / (tp + fn) if tp > 0 else 0


def accuracy(tp, fp, fn):
    return tp / (tp + fp + fn) if tp > 0 else 0


def f1(tp, fp, fn):
    return (2 * tp) / (2 * tp + fp + fn) if tp > 0 else 0


def _relabel(input):
    _, unique_labels = np.unique(input, return_inverse=True)
    return unique_labels.reshape(input.shape)


def _iou_matrix(gt, seg):
    # relabel gt and seg for smaller memory footprint of contingency table
    gt = _relabel(gt)
    seg = _relabel(seg)

    # get number of overlapping pixels between GT and SEG
    n_inter = contingency_table(gt, seg).A

    # number of pixels for GT instances
    n_gt = n_inter.sum(axis=1, keepdims=True)
    # number of pixels for SEG instances
    n_seg = n_inter.sum(axis=0, keepdims=True)

    # number of pixels in the union between GT and SEG instances
    n_union = n_gt + n_seg - n_inter

    iou_matrix = n_inter / n_union
    # make sure that the values are within [0,1] range
    assert 0 <= np.min(iou_matrix) <= np.max(iou_matrix) <= 1

    return iou_matrix


def _filter_instances(input, min_instance_size):
    """
    Filters instances smaller than 'min_instance_size' by overriding them with 0-index
    :param input: input instance segmentation
    """
    if min_instance_size is not None:
        labels, counts = np.unique(input, return_counts=True)
        for label, count in zip(labels, counts):
            if count < min_instance_size:
                logger.info(f'Ignoring instance label: {label}, size: {count}')
                input[input == label] = 0
    return input


class SegmentationMetrics:
    """
    Computes precision, recall, accuracy, f1 score for a given ground truth and predicted segmentation.
    Contingency table for a given ground truth and predicted segmentation is computed eagerly upon construction
    of the instance of `SegmentationMetrics`.

    Args:
        gt (ndarray): ground truth segmentation
        seg (ndarray): predicted segmentation
    """

    def __init__(self, gt, seg):
        gt = _relabel(gt)
        seg = _relabel(seg)
        self.iou_matrix = _iou_matrix(gt, seg)

    def metrics(self, iou_threshold):
        """
        Computes precision, recall, accuracy, f1 score at a given IoU threshold
        """
        # ignore background
        iou_matrix = self.iou_matrix[1:, 1:]
        detection_matrix = (iou_matrix > iou_threshold).astype(np.uint8)
        n_gt, n_seg = detection_matrix.shape

        # if the iou_matrix is empty or all values are 0
        trivial = min(n_gt, n_seg) == 0 or np.all(detection_matrix == 0)
        if trivial:
            tp = fp = fn = 0
        else:
            # count non-zero rows to get the number of TP
            tp = np.count_nonzero(detection_matrix.sum(axis=1))
            # count zero rows to get the number of FN
            fn = n_gt - tp
            # count zero columns to get the number of FP
            fp = n_seg - np.count_nonzero(detection_matrix.sum(axis=0))

        return {
            'precision': precision(tp, fp, fn),
            'recall': recall(tp, fp, fn),
            'accuracy': accuracy(tp, fp, fn),
            'f1': f1(tp, fp, fn)
        }


class Accuracy:
    """
    Computes accuracy between ground truth and predicted segmentation a a given threshold value.
    Defined as: AC = TP / (TP + FP + FN).
    Kaggle DSB2018 calls it Precision, see:
    https://www.kaggle.com/stkbailey/step-by-step-explanation-of-scoring-metric.
    """

    def __init__(self, iou_threshold):
        self.iou_threshold = iou_threshold

    def __call__(self, input_seg, gt_seg):
        metrics = SegmentationMetrics(gt_seg, input_seg).metrics(self.iou_threshold)
        return metrics['accuracy']


class AveragePrecision:
    """
    Average precision taken for the IoU range (0.5, 0.95) with a step of 0.05 as defined in:
    https://www.kaggle.com/stkbailey/step-by-step-explanation-of-scoring-metric
    """

    def __init__(self, out_precision_recall=None):
        self.iou_range = np.linspace(0.50, 0.95, 10)
        self.out_precision_recall = out_precision_recall

    def __call__(self, input_seg, gt_seg):
        logger.info('Computing contingency table...')
        # compute contingency_table
        sm = SegmentationMetrics(gt_seg, input_seg)
        # compute accuracy for each threshold
        acc = []
        pr_rec = []
        for iou in self.iou_range:
            metrics = sm.metrics(iou)
            logger.info(f"IoU: {iou}. Metrics: {metrics}")
            acc.append(metrics['accuracy'])
            pr_rec.append((metrics['recall'], metrics['precision']))

        self._save_precision_recall(pr_rec)
        # return the average
        return np.mean(acc)

    def _save_precision_recall(self, precision_recall):
        plt.figure(figsize=(20, 20))
        recall, precision = zip(*precision_recall)

        logger.info(f'Recall values: {recall}. Precision values: {precision}')

        plt.xlabel('Recall')
        plt.ylabel('Precision')

        plt.xlim(0, 1)
        plt.ylim(0, 1)

        plt.plot(recall, precision, '-ok')

        plt.savefig(self.out_precision_recall)


def main():
    parser = argparse.ArgumentParser(description='Validation metrics')
    parser.add_argument('--gt', type=str, required=True, nargs='+',
                        help="path to the ground truth file containing the 'label' dataset")
    parser.add_argument('--gt-dataset', type=str, required=False, default='label',
                        help="path to the ground truth file containing the 'label' dataset")

    parser.add_argument('--seg', type=str, required=True, nargs='+',
                        help="path to the segmentation file containing the 'segmentation' dataset")
    parser.add_argument('--seg-dataset', type=str, required=False, default='segmentation',
                        help="path to the segmentation file containing the 'segmentation' dataset")

    parser.add_argument('--min-size', type=int, required=False, default=500,
                        help="Minimum instance size to be considered for AveragePrecision score")

    args = parser.parse_args()

    for gt_f, seg_f in zip(args.gt, args.seg):
        logger.info(f'Running the metrics... Segmentation file: {seg_f}, ground truth: {gt_f}')
        with h5py.File(gt_f, 'r') as f:
            gt = f[args.gt_dataset][...]

        with h5py.File(seg_f, 'r') as f:
            seg = f[args.seg_dataset][...]

        ## filter small instances
        logger.info(f'Filtering ground truth instances smaller than: {args.min_size}')
        gt = _filter_instances(gt, args.min_size)
        logger.info(f'Filtering segmentation instances smaller than: {args.min_size}')
        seg = _filter_instances(seg, args.min_size)

        ## output path for the precision-recall curve
        out_path = os.path.splitext(seg_f)[0] + '_precision-recall.png'
        ap = AveragePrecision(out_path)

        ap = ap(seg, gt)
        logger.info(f'Average Precision score: {ap}')


if __name__ == '__main__':
    main()
