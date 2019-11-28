# import the segmentation functionality from elf
import h5py
import numpy as np
import elf.segmentation.mutex_watershed as mws


in_file = '/g/kreshuk/wolny/Datasets/Vladyslav/GT/GT_Ab1_test1_predictions_affinities.h5'
out_file = '/g/kreshuk/wolny/Datasets/Vladyslav/GT/GT_Ab1_test1_predictions_mws.h5'

print('Loading affinities...')
with h5py.File(in_file, 'r') as f:
    # load the affinities data
    affs = f['predictions'][2:]

offsets = [
    [-1, 0, 0], [0, -3, 0], [0, 0, -3],
    [-2, 0, 0], [0, -9, 0], [0, 0, -9],
    [-3, 0, 0], [0, -18, 0], [0, 0, -18]
]

strides = [1, 6, 6]
randomize_strides = True

print('Executing MWS...')
segmentation = mws.mutex_watershed(affs, offsets, strides, randomize_strides=randomize_strides)

segmentation = segmentation.astype('uint16')

print('Saving results...')
with h5py.File(out_file, 'w') as f:
    f.create_dataset('mws', data=segmentation, compression='gzip')
