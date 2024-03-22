

import os

def create_folder(folder):
    if os.path.exists(folder):
        assert os.path.isdir(folder), 'it exists but is not a folder'
    else:
        os.makedirs(folder)