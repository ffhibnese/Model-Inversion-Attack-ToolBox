import sys

sys.path.append('../../../src')

from modelinversion.datasets.preprocess import preprocess_ffhq64

if __name__ == '__main__':

    src_path = '<fill it>'
    dst_path = '<fill it>'

    preprocess_ffhq64(src_path, dst_path)
