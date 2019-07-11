import h5py
import numpy as np

element_size_um = np.array([1., 0.17297205, 0.17297205])
in_path = '/home/adrian/workspace/ilastik-datasets/Vladyslav/vlad_nuclei_predictions.h5'
out_path = '/home/adrian/workspace/ilastik-datasets/Vladyslav/vlad_nuclei_segm.h5'
with h5py.File(in_path, 'r+') as f:
    preds = f['predictions'][0]

    thresholds = [0.5, 0.8, 0.9]
    for t in thresholds:
        mask = preds > t
        mask = mask.astype(np.uint8)
        ds = f.create_dataset('nuclei_segm_' + str(t), data=mask, compression='gzip')
        ds.attrs['element_size_um'] = element_size_um
