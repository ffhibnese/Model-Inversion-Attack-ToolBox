import os

if __name__ == '__main__':
    os.system('rm -rf ./cache/*')
    result_dir = './results'
    for dir_name in os.listdir(result_dir):
        remove_dir = os.path.join(result_dir, dir_name)
        if os.path.isdir(remove_dir):
            os.system(f'rm -rf {remove_dir}')