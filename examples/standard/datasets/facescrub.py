import sys
sys.path.append('../../../src')

from modelinversion.datasets.preprocess import preprocess_facescrub

if __name__ == '__main__':
    
    src_path = '<fill it>'
    dst_path = '<fill it>'
    
    preprocess_facescrub(src_path, dst_path)