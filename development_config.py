import os

work_root_dir = './cache'
result_root_dir = './results'
ckpt_dir = './checkpoints'
dataset_dir = './datasets'

def get_dirs(method_name):
    work_dir = os.path.join(work_root_dir, method_name)
    result_dir = os.path.join(result_root_dir, method_name)
    
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    
    return {'work_dir': work_dir, 'result_dir': result_dir, 'ckpt_dir': ckpt_dir, 'dataset_dir': dataset_dir}