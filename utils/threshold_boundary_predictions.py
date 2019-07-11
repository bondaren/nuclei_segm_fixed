import h5py
import numpy as np

element_size_um = np.array([1., 0.17297205, 0.17297205])
file_path = '/home/adrian/workspace/ilastik-datasets/Vladyslav/vlad_boundary_predictions.h5'
with h5py.File(file_path, 'r+') as f:
    preds = f['predictions'][...]
    boundary = preds[0]

    thresholds = [0.5, 0.65, 0.8]
    for t in thresholds:
        mask = boundary > t
        mask = mask.astype(np.uint8)
        ds = f.create_dataset('threshold_' + str(t), data=mask, compression='gzip')
        ds.attrs['element_size_um'] = element_size_um
