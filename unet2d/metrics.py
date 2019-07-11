import importlib

import torch

from unet2d.utils import get_logger

LOGGER = get_logger('EvalMetric')


def compute_per_channel_dice(input, target, epsilon=1e-5, ignore_index=None, weight=None):
    # assumes that input is a normalized probability

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # mask ignore_index if present
    if ignore_index is not None:
        mask = target.clone().ne_(ignore_index)
        mask.requires_grad = False

        input = input * mask
        target = target * mask

    input = flatten(input)
    target = flatten(target)

    target = target.float()
    # Compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    denominator = (input + target).sum(-1)
    return 2. * intersect / denominator.clamp(min=epsilon)


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, H, W) -> (C, N * * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, H, W) -> (C, N, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, H, W) -> (C, N *  H * W)
    return transposed.view(C, -1)


class DiceCoefficient:
    """Computes Dice Coefficient.
    Generalized to multiple channels by computing per-channel Dice Score
    (as described in https://arxiv.org/pdf/1707.03237.pdf) and theTn simply taking the average.
    Input is expected to be probabilities instead of logits.
    This metric is mostly useful when channels contain the same semantic class (e.g. affinities computed with different offsets).
    DO NOT USE this metric when training with DiceLoss, otherwise the results will be biased towards the loss.
    """

    def __init__(self, epsilon=1e-5, ignore_index=None, **kwargs):
        self.epsilon = epsilon
        self.ignore_index = ignore_index

    def __call__(self, input, target):
        """
        :param input: 5D probability maps torch tensor (NxCxDxHxW)
        :param target: 4D or 5D ground truth torch tensor. 4D (NxDxHxW) tensor will be expanded to 5D as one-hot
        :return: Soft Dice Coefficient averaged over all channels/classes
        """
        # Average across channels in order to get the final score
        return torch.mean(compute_per_channel_dice(input, target, epsilon=self.epsilon, ignore_index=self.ignore_index))


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxHxW label image to NxCxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 3D input image (NxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 4D output image (NxCxHxW)
    """
    assert input.dim() == 3

    shape = input.size()
    shape = list(shape)
    shape.insert(1, C)
    shape = tuple(shape)

    src = input.unsqueeze(0)

    if ignore_index is not None:
        # create ignore_index mask for the result
        expanded_src = src.expand(shape)
        mask = expanded_src == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        src = src.clone()
        src[src == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, src, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, src, 1)


class MeanIoU:
    """
    Computes IoU for each class separately and then averages over all classes.
    """

    def __init__(self, skip_channels=(), ignore_index=None, **kwargs):
        """
        :param skip_channels: list/tuple of channels to be ignored from the IoU computation
        :param ignore_index: id of the label to be ignored from IoU computation
        """
        self.ignore_index = ignore_index
        self.skip_channels = skip_channels

    def __call__(self, input, target):
        """
        :param input: 4D probability maps torch float tensor (NxCxHxW)
        :param target: 3D or 4D ground truth torch tensor. 3D (NxHxW) tensor will be expanded to 4D as one-hot
        :return: intersection over union averaged over all channels
        """
        assert input.dim() == 4

        n_classes = input.size()[1]
        if target.dim() == 3:
            target = expand_as_one_hot(target, C=n_classes, ignore_index=self.ignore_index)

        # batch dim must be 1
        input = input[0]
        target = target[0]
        assert input.size() == target.size()

        binary_prediction = self._binarize_predictions(input, n_classes)

        if self.ignore_index is not None:
            # zero out ignore_index
            mask = target == self.ignore_index
            binary_prediction[mask] = 0
            target[mask] = 0

        # convert to uint8 just in case
        binary_prediction = binary_prediction.byte()
        target = target.byte()

        per_channel_iou = []
        for c in range(n_classes):
            if c in self.skip_channels:
                continue

            per_channel_iou.append(self._jaccard_index(binary_prediction[c], target[c]))

        assert per_channel_iou, "All channels were ignored from the computation"
        return torch.mean(torch.tensor(per_channel_iou))

    def _binarize_predictions(self, input, n_classes):
        """
        Puts 1 for the class/channel with the highest probability and 0 in other channels. Returns byte tensor of the
        same size as the input tensor.
        """
        if n_classes == 1:
            # for single channel input just threshold the probability map
            result = input > 0.5
            return result.long()

        _, max_index = torch.max(input, dim=0, keepdim=True)
        return torch.zeros_like(input, dtype=torch.uint8).scatter_(0, max_index, 1)

    def _jaccard_index(self, prediction, target, epsilon=1e-5):
        """
        Computes IoU for a given target and prediction tensors
        """
        return torch.sum(prediction & target).float() / torch.sum(prediction | target).float().clamp(min=epsilon)


def get_evaluation_metric(config):
    """
    Returns the evaluation metric function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'eval_metric' key
    :return: an instance of the evaluation metric
    """

    def _metric_class(class_name):
        m = importlib.import_module('unet2d.metrics')
        clazz = getattr(m, class_name)
        return clazz

    assert 'eval_metric' in config, 'Could not find evaluation metric configuration'
    metric_config = config['eval_metric']
    metric_class = _metric_class(metric_config['name'])
    return metric_class(**metric_config)
