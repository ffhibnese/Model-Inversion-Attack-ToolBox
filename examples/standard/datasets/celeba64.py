import sys

sys.path.append('../../../src')

from modelinversion.datasets.preprocess import preprocess_celeba64

if __name__ == '__main__':

    src_path = '<fill it>'
    dst_path = '<fill it>'
    split_files_path = '<fill it>'

    preprocess_celeba64(src_path, dst_path, split_files_path)
