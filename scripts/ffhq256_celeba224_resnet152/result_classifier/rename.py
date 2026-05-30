import os

for folder in os.listdir('.'):
    if 'neck' in folder:
        continue
    if os.path.isdir(folder):
        os.system(f"mv {folder} {folder.replace('facescrub', 'celeba')}")

for folder in os.listdir('.'):
    if 'neck' in folder or not os.path.isdir(folder):
        continue
    for file in os.listdir(folder):
        if file.endswith(".pth"):
            try:
                os.system(f"mv {folder}/{file} {folder}/{file.replace('facescrub', 'celeba')}")
            except:
                pass