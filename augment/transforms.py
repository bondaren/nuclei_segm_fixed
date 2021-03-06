import importlib

import numpy as np
import torch
from scipy.ndimage import rotate, map_coordinates, gaussian_filter
from skimage.filters import gaussian
from torchvision.transforms import Compose


class RandomFlip:
    """
    Randomly flips the image across the given axes. Image can be either 2D (HxW) or 3D (CxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """

    def __init__(self, random_state, **kwargs):
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state
        self.axes = (0, 1)

    def __call__(self, m):
        assert m.ndim in [2, 3], 'Supports only 2D or 3D images'

        for axis in self.axes:
            if self.random_state.uniform() > 0.5:
                if m.ndim == 2:
                    m = np.flip(m, axis)
                else:
                    channels = [np.flip(m[c], axis) for c in range(m.shape[0])]
                    m = np.stack(channels, axis=0)

        return m


class RandomRotate:
    """
    Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation axis is picked at random from the list of provided axes.
    """

    def __init__(self, random_state, angle_spectrum=10, axes=None, mode='constant', order=0, **kwargs):
        self.random_state = random_state
        self.angle_spectrum = angle_spectrum
        self.mode = mode
        self.order = order

    def __call__(self, m):
        angle = self.random_state.randint(-self.angle_spectrum, self.angle_spectrum)

        if m.ndim == 2:
            m = rotate(m, angle, axes=(1, 0), reshape=False, order=self.order, mode=self.mode, cval=-1)
        else:
            channels = [rotate(m[c], angle, reshape=False, order=self.order, mode=self.mode, cval=-1) for c
                        in range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m


class RandomContrast:
    """
        Adjust the brightness of an image by a random factor.
    """

    def __init__(self, random_state, factor=0.5, execution_probability=0.1, **kwargs):
        self.random_state = random_state
        self.factor = factor
        self.execution_probability = execution_probability

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            brightness_factor = self.factor + self.random_state.uniform()
            return np.clip(m * brightness_factor, 0, 1)

        return m


# it's relatively slow, i.e. ~1s per patch of size 64x200x200, so use multiple workers in the DataLoader
# remember to use spline_order=3 when transforming the labels
class ElasticDeformation:
    """
    Apply elasitc deformations of 3D patches on a per-voxel mesh. Assumes ZYX axis order!
    Based on: https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62
    """

    def __init__(self, random_state, spline_order, alpha=15, sigma=3, execution_probability=0.3, **kwargs):
        """
        :param spline_order: the order of spline interpolation (use 0 for labeled images)
        :param alpha: scaling factor for deformations
        :param sigma: smoothing factor for Gaussian filter
        """
        self.random_state = random_state
        self.spline_order = spline_order
        self.alpha = alpha
        self.sigma = sigma
        self.execution_probability = execution_probability

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            assert m.ndim == 2
            dy = gaussian_filter(self.random_state.randn(*m.shape), self.sigma, mode="constant", cval=0) * self.alpha
            dx = gaussian_filter(self.random_state.randn(*m.shape), self.sigma, mode="constant", cval=0) * self.alpha

            y_dim, x_dim = m.shape
            y, x = np.meshgrid(np.arange(y_dim), np.arange(x_dim), indexing='ij')
            indices = y + dy, x + dx
            return map_coordinates(m, indices, order=self.spline_order, mode='reflect')

        return m


def blur_boundary(boundary, sigma):
    boundary = gaussian(boundary, sigma=sigma)
    boundary[boundary >= 0.5] = 1
    boundary[boundary < 0.5] = 0
    return boundary


class Normalize:
    """
    Normalizes a given input tensor to be 0-mean and 1-std.
    mean and std parameter have to be provided explicitly.
    """

    def __init__(self, mean, std, eps=1e-4, **kwargs):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, m):
        return (m - self.mean) / (self.std + self.eps)


class RangeNormalize:
    def __init__(self, max_value=255, **kwargs):
        self.max_value = max_value

    def __call__(self, m):
        return m / self.max_value


class GaussianNoise:
    def __init__(self, random_state, max_sigma, max_value=255, **kwargs):
        self.random_state = random_state
        self.max_sigma = max_sigma
        self.max_value = max_value

    def __call__(self, m):
        # pick std dev from [0; max_sigma]
        std = self.random_state.randint(self.max_sigma)
        gaussian_noise = self.random_state.normal(0, std, m.shape)
        noisy_m = m + gaussian_noise
        return np.clip(noisy_m, 0, self.max_value).astype(m.dtype)


class ToTensor:
    """
    Converts a given input numpy.ndarray into torch.Tensor. Adds additional 'channel' axis when the input is 3D
    and expand_dims=True (use for raw data of the shape (D, H, W)).
    """

    def __init__(self, expand_dims, dtype=np.float32, **kwargs):
        self.expand_dims = expand_dims
        self.dtype = dtype

    def __call__(self, m):
        assert m.ndim in [2, 3], 'Supports only 2D or 3D images'
        # add channel dimension
        if self.expand_dims and m.ndim == 2:
            m = np.expand_dims(m, axis=0)

        return torch.from_numpy(m.astype(dtype=self.dtype))


class Identity:
    def __call__(self, m):
        return m


def get_transformer(config, mean, std, phase):
    if phase == 'val':
        phase = 'test'

    assert phase in config, f'Cannot find transformer config for phase: {phase}'
    phase_config = config[phase]
    return Transformer(phase_config, mean, std)


class Transformer:
    def __init__(self, phase_config, mean, std):
        self.phase_config = phase_config
        self.config_base = {'mean': mean, 'std': std}
        self.seed = 47

    def raw_transform(self):
        return self._create_transform('raw')

    def label_transform(self):
        return self._create_transform('label')

    def weight_transform(self):
        return self._create_transform('weight')

    @staticmethod
    def _transformer_class(class_name):
        m = importlib.import_module('augment.transforms')
        clazz = getattr(m, class_name)
        return clazz

    def _create_transform(self, name):
        assert name in self.phase_config, f'Could not find {name} transform'
        return Compose([
            self._create_augmentation(c) for c in self.phase_config[name]
        ])

    def _create_augmentation(self, c):
        config = dict(self.config_base)
        config.update(c)
        config['random_state'] = np.random.RandomState(self.seed)
        aug_class = self._transformer_class(config['name'])
        return aug_class(**config)


def _recover_ignore_index(input, orig, ignore_index):
    if ignore_index is not None:
        mask = orig == ignore_index
        input[mask] = ignore_index

    return input


from scipy.ndimage.filters import convolve

class AbstractLabelToBoundary:
    AXES_TRANSPOSE = [
        (0, 1),  # X
        (1, 0)  # Y
    ]

    def __init__(self, ignore_index=None, append_label=False, **kwargs):
        """
        :param ignore_index: label to be ignored in the output, i.e. after computing the boundary the label ignore_index
            will be restored where is was in the patch originally
        :param append_label: if True append the orignal ground truth labels to the last channel
        :param sigma: standard deviation for Gaussian kernel
        """
        self.ignore_index = ignore_index
        self.append_label = append_label

    def __call__(self, m):
        """
        Extract boundaries from a given 2D label tensor.
        :param m: input 2D tensor
        :return: binary mask, with 1-label corresponding to the boundary and 0-label corresponding to the background
        """
        assert m.ndim == 2

        kernels = self.get_kernels()
        boundary_arr = [np.where(np.abs(convolve(m, kernel)) > 0, 1, 0) for kernel in kernels]
        channels = np.stack(boundary_arr)
        results = [_recover_ignore_index(channels[i], m, self.ignore_index) for i in range(channels.shape[0])]

        if self.append_label:
            # append original input data
            results.append(m)

        # stack across channel dim
        return np.stack(results, axis=0)

    @staticmethod
    def create_kernel(axis, offset):
        # create conv kernel
        k_size = offset + 1
        k = np.zeros((1, k_size), dtype=np.int)
        k[0, 0] = 1
        k[0, offset] = -1
        return np.transpose(k, axis)

    def get_kernels(self):
        raise NotImplementedError


class LabelToAffinities(AbstractLabelToBoundary):
    """
    Converts a given volumetric label array to binary mask corresponding to borders between labels (which can be seen
    as an affinity graph: https://arxiv.org/pdf/1706.00120.pdf)
    One specify the offsets (thickness) of the border. The boundary will be computed via the convolution operator.
    """

    def __init__(self, offsets, ignore_index=None, append_label=False, **kwargs):
        super().__init__(ignore_index=ignore_index, append_label=append_label)

        assert isinstance(offsets, list) or isinstance(offsets, tuple), 'offsets must be a list or a tuple'
        assert all(a > 0 for a in offsets), "'offsets must be positive"
        assert len(set(offsets)) == len(offsets), "'offsets' must be unique"

        self.kernels = []
        # create kernel for every axis-offset pair
        for xy_offset in offsets:
            for axis_ind, axis in enumerate(self.AXES_TRANSPOSE):
                # create kernels for a given offset in every direction
                self.kernels.append(self.create_kernel(axis, xy_offset))

    def get_kernels(self):
        return self.kernels



