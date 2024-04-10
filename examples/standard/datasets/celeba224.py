import sys

sys.path.append('../../../src')

from modelinversion.datasets.preprocess import preprocess_celeba224

if __name__ == '__main__':

    src_path = '<fill it>'
    dst_path = '<fill it>'
    split_files_path = '<fill it>'
    mode = 'copy'

    preprocess_celeba224(src_path, dst_path, split_files_path, mode=mode)
