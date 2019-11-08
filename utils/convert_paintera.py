import glob
import os

import h5py
import z5py

from paintera_tools import convert_to_paintera_format
from paintera_tools import set_default_shebang


def convert_cremi(input_path, tmp_folder):
    # NOTE: the raw data needs to be multiscale, i.e. 'raw_key' needs to be a group with
    # datasets 'raw_key/s0' ... 'raw_key/sN'. It's ok if there's only a single scale, i.e. 'raw_key/s0'
    raw_key = 'raw'

    ass_path = ass_key = ''
    in_key = 'label'

    # output key: we stote the new paintera dataset here
    out_key = 'paintera'

    # shebang to environment with all necessary dependencies
    shebang = '#! /home/adrian/miniconda3/envs/cluster_env/bin/python'
    set_default_shebang(shebang)

    res = [0.98, 0.19, 0.19]
    convert_to_paintera_format(input_path, raw_key, in_key, out_key,
                               label_scale=0, resolution=res,
                               tmp_folder=tmp_folder, target='local', max_jobs=4, max_threads=8,
                               assignment_path=ass_path, assignment_key=ass_key,
                               label_block_mapping_compression='raw')


if __name__ == '__main__':
    input_files = '/home/adrian/workspace/ilastik-datasets/Vladyslav/GT/*.h5'
    for t, input_path in enumerate(glob.glob(input_files)):
        output_path = os.path.splitext(input_path)[0] + '.n5'

        print(f'Loading {input_path}...')
        with h5py.File(input_path, 'r') as f:
            gt = f['dt_watershed_2'][...]
            raw = f['raw'][...]

        print(f'Saving {output_path}...')
        with z5py.File(output_path) as f:
            label_ds = f.create_dataset('label', data=gt.astype('uint64'), compression='gzip', chunks=(64, 64, 64))
            label_ds.attrs['maxId'] = int(gt.max())
            f.create_dataset('raw/s0', data=raw, compression='gzip', chunks=(64, 64, 64))

        tmp_dir = f'./tmp_convert_{t}'
        convert_cremi(output_path, tmp_dir)
