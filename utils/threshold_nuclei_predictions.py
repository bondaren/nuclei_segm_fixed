import os
import glob
import h5py
import numpy as np
from skimage import measure

element_size_um = np.array([1., 0.17297205, 0.17297205])

for in_path in glob.glob('/g/kreshuk/wolny/Datasets/Vladyslav/antibody/*predictions.h5'):
    with h5py.File(in_path, 'r') as f:
        print(f'Processing {in_path}...')

        preds = f['predictions'][0]

        out_path = os.path.splitext(in_path)[0] + '_segm.h5'

        with h5py.File(out_path, 'w') as g:
            thresholds = [0.5, 0.65, 0.8]
            for t in thresholds:
                print(f'Processing threshold {t}...')
                mask = preds > t
                mask = mask.astype(np.uint8)
                ds = g.create_dataset('nuclei_mask_' + str(t), data=mask, compression='gzip')
                ds.attrs['element_size_um'] = element_size_um

                instance_segm = measure.label(mask, connectivity=1)
                instance_segm = instance_segm.astype('uint16')
                ds = g.create_dataset('nuclei_segm_' + str(t), data=instance_segm, compression='gzip')
                ds.attrs['element_size_um'] = element_size_um
