# import the segmentation functionality from elf
import h5py
import numpy as np
from affogato.segmentation import compute_mws_segmentation



def mutex_watershed(affs, offsets, strides,
                    randomize_strides=False, mask=None,
                    noise_level=0):
    """ Compute mutex watershed segmentation.

    Introduced in "The Mutex Watershed and its Objective: Efficient, Parameter-Free Image Partitioning":
    https://arxiv.org/pdf/1904.12654.pdf

    Arguments:
        affs [np.ndarray] - input affinity map
        offsets [list[list[int]]] - pixel offsets corresponding to affinity channels
        strides [list[int]] - strides used to sub-sample long range edges
        randomize_strides [bool] - randomize the strides? (default: False)
        mask [np.ndarray] - mask to exclude from segmentation (default: None)
        noise_level [float] - sigma of noise added to affinities (default: 0)
    """
    ndim = len(offsets[0])
    if noise_level > 0:
        affs += noise_level * np.random.rand(*affs.shape)
    affs[:ndim] *= -1
    affs[:ndim] += 1
    seg = compute_mws_segmentation(affs, offsets,
                                   number_of_attractive_channels=ndim,
                                   strides=strides, mask=mask,
                                   randomize_strides=randomize_strides)
    #relabelConsecutive(seg, out=seg, start_label=1, keep_zeros=mask is not None)
    return seg

in_file = '/g/kreshuk/wolny/Datasets/Vladyslav/GT/GT_Ab1_test1_predictions_affinities.h5'
out_file = '/g/kreshuk/wolny/Datasets/Vladyslav/GT/GT_Ab1_test1_predictions_mws.h5'

print('Loading affinities...')
with h5py.File(in_file, 'r') as f:
    # load the affinities data
    affs = f['predictions'][2:]

offsets = [
    [-1, 0, 0], [0, -1, 0], [0, 0, -1],
    [-2, 0, 0], [0, -9, 0], [0, 0, -9],
    [-3, 0, 0], [0, -18, 0], [0, 0, -18]
]

strides = [1, 6, 6]
randomize_strides = True

print('Executing MWS...')
segmentation = mutex_watershed(affs, offsets, strides, randomize_strides=randomize_strides)

segmentation = segmentation.astype('uint16')

print('Saving results...')
with h5py.File(out_file, 'w') as f:
    f.create_dataset('mws', data=segmentation, compression='gzip')
