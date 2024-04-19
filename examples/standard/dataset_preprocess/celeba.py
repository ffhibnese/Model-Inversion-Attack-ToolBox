import sys

sys.path.append('../../../src')

from modelinversion.datasets.preprocess import preprocess_celeba

if __name__ == '__main__':

    # '<fill it>'
    src_path = '../../../../CelebA'
    dst_path = '../../../test/celeba'
    split_files_path = '../../../src/modelinversion/datasets/split_files'
    mode = 'copy'

    preprocess_celeba(src_path, dst_path, split_files_path, mode=mode)
