import os
import argparse


def split(raw_img_dir, split_file_dir, dst_dir):
    with open(split_file_dir) as f:
        data = f.readlines()
    for s in data:
        s = s.strip()
        if s != '':
            s, label = s.split(' ')
            
            src_path = os.path.join(raw_img_dir, f'{s}')
            dst_label_dir = os.path.join(dst_dir, f'{label}')
            os.makedirs(dst_label_dir, exist_ok=True)
            dst_path = os.path.join(dst_label_dir, s)
            
            os.system(f'cp {src_path} {dst_path}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_dir', type=str)
    
    args = parser.parse_args()
    
    # raw_img_dir = './raw/celeba/img_align_celeba'
    raw_img_dir = args.file_dir
    
    split_files = [
        './split_files/private_train.txt',
        './split_files/private_test.txt',
        './split_files/public.txt'
    ]
    
    dst_dirs = [
        './split/private/train',
        './split/private/test',
        './split/public'
    ]
    print(os.path.abspath('.'))
    # split_file_dir = './split_files/private_train.txt'
    # dst_dir = './split/private_re_idx/train'
    for dst_dir, split_file_dir in zip(dst_dirs, split_files):
        split(raw_img_dir, split_file_dir, dst_dir)