import os

if __name__ == "__main__":
    datasets_path = "../data/FaceScrub/facescrub/train/"
    types_name = os.listdir(datasets_path)
    types_name = sorted(types_name)

    list_file = open('facescrub_train.txt', 'w')
    for cls_id, type_name in enumerate(types_name):
        print(cls_id, type_name)
        photos_path = os.path.join(datasets_path, type_name)
        if not os.path.isdir(photos_path):
            continue
        photos_name = os.listdir(photos_path)

        for photo_name in photos_name:
            if str(photo_name) == '.DS_Store':
                continue

            # types_name不存在 .DS_store 文件，所以不-1
            list_file.write(
                str(cls_id)
                + ";"
                + '%s'
                % (os.path.join(os.path.abspath(datasets_path), type_name, photo_name))
            )

            # types_name存在 .DS_store 文件，所以 -1
            # list_file.write(str(cls_id-1) + ";" + '%s'%(os.path.join(os.path.abspath(datasets_path), type_name, photo_name)))
            list_file.write('\n')
    list_file.close()
