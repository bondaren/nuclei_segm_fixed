# import the segmentation functionality from elf
import argparse
import os
import h5py
import numpy as np
from affogato.segmentation import compute_mws_segmentation
from elf.segmentation.watershed import apply_size_filter


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
    # relabelConsecutive(seg, out=seg, start_label=1, keep_zeros=mask is not None)
    return seg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MWS seg')
    parser.add_argument('--pmaps', type=str, required=True, help='path to the network predictions')
    args = parser.parse_args()

    in_file = args.pmaps

    out_file = os.path.splitext(in_file)[0] + '_mws.h5'

    print('Loading affinities...')
    with h5py.File(in_file, 'r') as f:
        mask = f['predictions'][0]
        mask = mask > 0.5
        mask = np.logical_and(mask)

        pmaps = f['predictions'][1]
        pmaps = pmaps.astype('float32')
        # load the affinities data
        affs = f['predictions'][2:]

    offsets = [
        [-1, 0, 0], [0, -1, 0], [0, 0, -1],
        [-2, 0, 0], [0, -3, 0], [0, 0, -3],
        [-3, 0, 0], [0, -9, 0], [0, 0, -9],
        [-4, 0, 0], [0, -18, 0], [0, 0, -18]
    ]

    strides = [1, 6, 6]
    randomize_strides = True

    print('Executing MWS...')
    segmentation = mutex_watershed(affs, offsets, strides, randomize_strides=randomize_strides, mask=mask)

    print('Filtering small objects...')
    segmentation = segmentation.astype('uint32')
    segmentation, _ = apply_size_filter(segmentation, pmaps, size_filter=500)
    segmentation = segmentation.astype('uint16')

    print('Saving results to:', out_file)
    with h5py.File(out_file, 'w') as f:
        f.create_dataset('segmentation', data=segmentation, compression='gzip')
