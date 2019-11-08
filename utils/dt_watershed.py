from scipy import ndimage as ndi

from scipy.ndimage import gaussian_filter
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import measure

import h5py
import numpy as np
import glob

for in_file in glob.glob('/home/adrian/workspace/ilastik-datasets/Vladyslav/GT/*.h5'):
    print(f'Processing {in_file}...')

    with h5py.File(in_file, 'r+') as f:
        label = f['label'][...]
        if 'dtw' in f:
            del f['dtw']

        print('Distance transform...')
        # Generate the markers as local maxima of the distance to the background
        distance = ndi.distance_transform_edt(label)
        # Smooth DT
        distance = gaussian_filter(distance, sigma=2.0)

        footprints = [(10, 30, 30), (5, 25, 25), (5, 10, 10)]
        for i, fp in enumerate(footprints):
            print(f'Footprint: {fp}')
            print('Finding local maxima...')
            # Find local maxima
            local_maxi = peak_local_max(distance, indices=False, footprint=np.ones(fp), labels=label)
            markers = ndi.label(local_maxi)[0]
            print('Watershed...')
            labels = watershed(-distance, markers, mask=label)

            if np.max(labels) > 65535:
                raise RuntimeError('Cannot convert to uint16')
            else:
                labels = labels.astype('uint16')

            f.create_dataset(f'dt_watershed_{i}', data=labels, compression='gzip')

        # save connected components
        cc = measure.label(label, connectivity=1)
        f.create_dataset('cc', data=cc.astype('uint16'), compression='gzip')
