#!/usr/bin/env python3
# coding=utf-8
import os
import subprocess


ALL_CACHE = {
    # conf_mask.pt
    'conf_mask.pt': 'https://purdue0-my.sharepoint.com/:u:/g/personal/an93_purdue_edu/EfGCarPZZ-BGmE4RnZpVze4BVogJVdI3K46JDJzJsqcU5g?e=evkn75&download=1',
    'stylegan_sample_z_stylegan_celeba_partial256_0.7_8_all_ws.pt': 'https://purdue0-my.sharepoint.com/:u:/g/personal/an93_purdue_edu/Eee2Fvs7269DoZ8bRVKbBjEBY6bi0z02eLc6ApOiTc-wwA?e=Bz7Zzy&download=1',
    # resnet50 on vggface2
    'resnet50_scratch_dag.pth': 'https://purdue0-my.sharepoint.com/:u:/g/personal/an93_purdue_edu/EZOXU_L8CQdHvWdWnRfV7F4BCE-JGamMjKYwBWuPk5pyVQ?e=Hmzguu&download=1',
    'blackbox_attack_data/stylegan/resnet50/no_dropout/all_logits.pt': 'https://purdue0-my.sharepoint.com/:u:/g/personal/an93_purdue_edu/EVqqJuJLAu5ElAlfikeYfjkBdL2P35AwPuNkth_GoPIZNA?e=vl320x&download=1',
    'centroid_data/resnet50/test/centroid_logits.pt': 'https://purdue0-my.sharepoint.com/:u:/g/personal/an93_purdue_edu/ETW3GDEXXMtAj16IGee3nmEBJ4CcUTcQ5GEsiZVd25FOXQ?e=kMdQHp&download=1',
    # vgg16 on vggface
    'vgg_face_dag.pth': 'https://purdue0-my.sharepoint.com/:u:/g/personal/an93_purdue_edu/EXTGdHk8fnZCriclmVjRFeIBif04VtUlKOYSFF9a1fh08A?e=sAaAQi&download=1',
    'blackbox_attack_data/stylegan/vgg16/no_dropout/all_logits.pt': 'https://purdue0-my.sharepoint.com/:u:/g/personal/an93_purdue_edu/ESV59GCFGT9KvzvZOkvToaABFKAIPv4_u5rj0Yg03lpjew?e=8g6DxL&download=1',
    'blackbox_attack_data/stylegan/vgg16/use_dropout/all_logits.pt': 'https://purdue0-my.sharepoint.com/:u:/g/personal/an93_purdue_edu/EV3a2youCpNNg4w4dAaIbsQBvcx53lcehHF-hV_2HlS2dA?e=8ECbQi&download=1',
    'centroid_data/vgg16/test/centroid_logits.pt': 'https://purdue0-my.sharepoint.com/:u:/g/personal/an93_purdue_edu/EZ-RTSyk3K9Gi2NndgYoTDkBRIRmIeVqn57jimZ-UHCChA?e=CuYwXa&download=1',
}


def download(filename, url):

    def create_folder(folder):
        if os.path.exists(folder):
            assert os.path.isdir(folder), 'it exists but is not a folder'
        else:
            os.makedirs(folder)

    if not os.path.exists(filename):
        print('Downloading', filename)
        dirname = os.path.dirname(filename)
        if dirname:
            create_folder(dirname)
        subprocess.call(['wget', '--quiet', '-O', filename, url])


def main():
    for filename, url in ALL_CACHE.items():
        download(filename, url)


if __name__ == '__main__':
    main()
