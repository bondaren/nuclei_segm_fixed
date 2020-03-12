import argparse
import os
import sys

import h5py
import z5py
from paintera_tools import convert_to_paintera_format
from paintera_tools import set_default_shebang, set_default_block_shape


def convert_cremi(input_path, with_assignments, tmp_folder, res, convert_to_label_multisets=False):
    """ Convert segmentation to paintera format.

    You can donwload the example data from
    https://drive.google.com/open?id=15hZmM4cu_H_ruhlgXilNWgDZWMpuo9XK

    Arguments:
        input_path [str]: path to n5 file with raw data and segmentation
        with_assignments [bool]: convert to format with / without fragment-segment assignment
        tmp_folder [str]: folder to store tempory data
        res [list]: voxel size (ZYX)
    """

    # NOTE: the raw data needs to be multiscale, i.e. 'raw_key' needs to be a group with
    # datasets 'raw_key/s0' ... 'raw_key/sN'. It's ok if there's only a single scale, i.e. 'raw_key/s0'
    # you can use 'paintera_tools.converter.downscale' to compute a scale pyramid for your raw data
    raw_key = 'raw'

    # we have two options: create a dataset with or without fragment-segment assignments
    if with_assignments:
        # with assignments: we need an assignment path and key
        # and we use the watersheds as input segmentation
        ass_path = input_path
        ass_key = 'node_labels'
        in_key = 'segmentation/watershed'
    else:
        # without assignments: no assignment path or key
        # and we use the multicut segmentation as input segmentation
        ass_path = ass_key = ''
        in_key = 'label'

    # output key: we store the new paintera dataset here
    out_key = 'paintera'

    # shebang to environment with all necessary dependencies
    shebang = f'#! {sys.executable}'
    set_default_shebang(shebang)
    set_default_block_shape([32, 256, 256])

    if convert_to_label_multisets:
        restrict_sets = [-1, -1, 5, 3]
    else:
        restrict_sets = []

    convert_to_paintera_format(input_path, raw_key, in_key, out_key,
                               label_scale=0, resolution=res,
                               tmp_folder=tmp_folder, target='local', max_jobs=4, max_threads=8,
                               assignment_path=ass_path, assignment_key=ass_key,
                               convert_to_label_multisets=convert_to_label_multisets,
                               restrict_sets=restrict_sets,
                               label_block_mapping_compression='raw', copy_labels=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert H5 to Paintera format')
    parser.add_argument('--file', type=str, required=True, help="Path to the H5 to be conveted")
    parser.add_argument('--vsize', type=float, nargs='+', help='Voxel size (ZYX)')
    parser.add_argument('--rawds', type=str, required=False, default='raw',
                        help="path to the raw dataset")
    parser.add_argument('--labelds', type=str, required=False, default='label',
                        help="path to the label dataset")

    args = parser.parse_args()

    res = args.vsize
    assert len(res) == 3, f'Incorrect voxel size: {res}'

    in_file = args.file
    n5_path = os.path.splitext(in_file)[0] + '.n5'

    print(f'H5 path: {in_file}')
    print(f'N5 path: {n5_path}')
    print(f'Res: {res}')

    with h5py.File(in_file, 'r') as g:
        raw = g[args.rawds][...]
        label = g[args.labelds][...]
        label = label.astype('uint64')

    with z5py.File(n5_path) as f:
        ds = f.create_dataset('label', data=label, compression='gzip', chunks=(64, 64, 64))
        ds.attrs['maxId'] = int(label.max())
        f.create_dataset('raw/s0', data=raw, compression='gzip', chunks=(64, 64, 64))

    tmp_folder = os.path.splitext(in_file)[0] + '_tmp_convert'
    with_assignments = False
    convert_cremi(n5_path, with_assignments, tmp_folder, res)
