import torch
from models.facenet import Facenet
from utils.train_facenet import get_num_classes
from utils.detect_crop_face import facenet_input_preprocessing, detect_crop_face
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm
import dlib
import numpy as np
import os


device        = 'cuda'
img_size      = 256
dataset       = 'facescrub'
t_model_bone  = 'mobile_net'
eva_backbone  = 'mobile_facenet'
facenet_path  = 'trained_models/FaceScrub-BackboneMobileFaceNet-Epoch4-Train_Acc0.992-Val_Acc0.971.pth'  # path to evaluation model
train_txt     = ''  # training data .txt (generate by annotation_face_train.py)
img_dir       = 'gen_figures/DE_facescrub_mobile_M100_counter'
detector = dlib.get_frontal_face_detector()
num_classes   = get_num_classes(train_txt)
face_shape    = [160, 160]
all_id        = 526

# 加载训练好的目标模型
test_model = Facenet(backbone=eva_backbone, num_classes=num_classes)
test_model.load_state_dict(torch.load(facenet_path, map_location='cpu'), strict=True)
test_model.to(device)
test_model.eval()

knn_sum    = 0
mn_sum     = 0

with open(train_txt,"r") as f:
     lines = f.readlines()
count = 0
c_max = len(lines)-1

for label in tqdm(range(all_id)):
    label_ = label
    img_path      = f'{img_dir}/de_label{label_}_best.jpg'
    img = Image.open(img_path)
    _, cropped = detect_crop_face(img, detector)
    face_input = facenet_input_preprocessing(cropped,(160,160)).to(device)

    tmp_dir = []
    ct2     = 0
    while int(lines[count].split(';')[0]) == label_:
        if ct2 < 100:
            tmp_dir.append((lines[count].split(';')[1]).split()[0])
            ct2 += 1
        count += 1
        if count > c_max:
            break

    n_img = len(tmp_dir)
    tmp_imgs = torch.zeros((n_img, 3, *face_shape)).to(device)
    for j in range(n_img):
        t_img = Image.open(tmp_dir[j])
        tmp_imgs[j:j+1] = facenet_input_preprocessing(t_img,(160,160))

    before_norm, _    = test_model.forward_feature(face_input)
    before_norm_cp, _ = test_model.forward_feature(tmp_imgs)
    dis_vecs          = before_norm_cp - before_norm

    t_norm  =  dis_vecs.norm(p=2, dim=1)
    knn_sum += (t_norm.min()).item()
    mn_sum  += (t_norm.mean()).item()


knn_avg = knn_sum / all_id
mn_avg  = mn_sum /all_id

print(f'{dataset} KNN Dist: {knn_avg};'
      f'{dataset} MN Dist: {mn_avg};')
