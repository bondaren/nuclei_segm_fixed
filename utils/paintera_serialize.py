import h5py
from skimage import measure

from paintera_tools import serialize_from_commit
from paintera_tools import set_default_shebang
from z5py.converter import convert_to_h5
import os


def serialize_cremi(path, tmp_folder):
    in_key = 'paintera'
    out_key = 'segmentation/corrected'

    shebang = '#! /home/adrian/miniconda3/envs/cluster_env/bin/python'
    set_default_shebang(shebang)

    serialize_from_commit(path, in_key, path, out_key,
                          tmp_folder, 1, 'local',
                          relabel_output=True)


if __name__ == '__main__':
    in_files = [
        '/home/adrian/workspace/ilastik-datasets/Vladyslav/GT/train/CT_Ab1_train.n5',
        '/home/adrian/workspace/ilastik-datasets/Vladyslav/GT/train/CT_Ab2_train.n5',
        '/home/adrian/workspace/ilastik-datasets/Vladyslav/GT/test/GT_Ab1_test.n5',
        '/home/adrian/workspace/ilastik-datasets/Vladyslav/GT/test/GT_Ab2_test.n5',
    ]
    for t, input_path in enumerate(in_files):
        print(f'Processing {input_path}...')
        # extract corrected segmantation
        tmp_dir = f'./tmp_serialize_{t}'
        serialize_cremi(input_path, tmp_dir)

        output_path = os.path.splitext(input_path)[0] + '.h5'
        convert_to_h5(input_path, output_path, 'segmentation/corrected', 'label64', n_threads=8, compression='gzip')

        with h5py.File(output_path, 'r+') as f:
            label64 = f['label64'][...]
            cc = measure.label(label64, connectivity=2)
            cc = cc.astype('uint16')
            f.create_dataset('label', data=cc, compression='gzip')

        convert_to_h5(input_path, output_path, 'raw/s0', 'raw', n_threads=8, compression='gzip')
