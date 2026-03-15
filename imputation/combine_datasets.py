import argparse
import numpy as np
import os


def main(base_path='imputation/data/'):

    all_train_notrevin_x = []
    all_train_revin_x = []

    all_val_notrevin_x = []
    all_val_revin_x = []

    all_test_notrevin_x = []
    all_test_revin_x = []

    for datatype in ['electricity/', 'ETTh1/', 'ETTh2/', 'ETTm1/', 'ETTm2/', 'weather/']:

        all_train_notrevin_x.append(np.load(base_path + datatype + 'train_notrevin_x.npy'))
        all_train_revin_x.append(np.load(base_path + datatype + 'train_revin_x.npy'))

        all_val_notrevin_x.append(np.load(base_path + datatype + 'val_notrevin_x.npy'))
        all_val_revin_x.append(np.load(base_path + datatype + 'val_revin_x.npy'))

        all_test_notrevin_x.append(np.load(base_path + datatype + 'test_notrevin_x.npy'))
        all_test_revin_x.append(np.load(base_path + datatype + 'test_revin_x.npy'))

    print('starting')

    all_train_notrevin_x_arr = np.concatenate(all_train_notrevin_x, axis=0)
    all_train_revin_x_arr = np.concatenate(all_train_revin_x, axis=0)

    print('train_done')

    all_val_notrevin_x_arr = np.concatenate(all_val_notrevin_x, axis=0)
    all_val_revin_x_arr = np.concatenate(all_val_revin_x, axis=0)

    print('val_done')

    all_test_notrevin_x_arr = np.concatenate(all_test_notrevin_x, axis=0)
    all_test_revin_x_arr = np.concatenate(all_test_revin_x, axis=0)

    print('test_done')

    os.makedirs(base_path + 'all', exist_ok=True)

    np.save(base_path + 'all/train_notrevin_x.npy', all_train_notrevin_x_arr)
    np.save(base_path + 'all/train_revin_x.npy', all_train_revin_x_arr)

    np.save(base_path + 'all/val_notrevin_x.npy', all_val_notrevin_x_arr)
    np.save(base_path + 'all/val_revin_x.npy', all_val_revin_x_arr)

    np.save(base_path + 'all/test_notrevin_x.npy', all_test_notrevin_x_arr)
    np.save(base_path + 'all/test_revin_x.npy', all_test_revin_x_arr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine individual dataset files into a single "all" dataset')
    parser.add_argument('--base_path', type=str, default='imputation/data/',
                        help='base path containing individual dataset folders')
    args = parser.parse_args()

    # Ensure trailing slash
    bp = args.base_path
    if not bp.endswith('/'):
        bp += '/'

    main(base_path=bp)
