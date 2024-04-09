import os

if __name__ == "__main__":
    # datasets_path   = "../data/CASIA-WebFaces/datasets/"
    datasets_path = "../data/FaceScrub/facescrub_manuclean/train/"
    types_name = os.listdir(datasets_path)
    types_name = sorted(types_name)

    list_file = open('dir2label-facescrub.txt', 'w')
    for cls_id, type_name in enumerate(types_name):
        list_file.write(type_name + ': ' + str(cls_id))
        list_file.write('\n')
    list_file.close()
