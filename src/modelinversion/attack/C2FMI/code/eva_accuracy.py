import torch
from models.facenet import Facenet

# from train_facenet import get_num_classes
from utils.detect_crop_face import facenet_input_preprocessing, detect_crop_face
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm
import dlib
import numpy as np

# 处理参数
device = 'cuda'
img_size = 256
dataset = 'facescrub'  # dataset
t_model_bone = 'mobile_net'  # backbone of target model
eva_backbone = 'mobile_facenet'  # backbone of evaluation model
facenet_path = 'trained_models/FaceScrub-BackboneMobileFaceNet-Epoch4-Train_Acc0.992-Val_Acc0.971.pth'  # path to evaluation model
num_classes = 526  # total classes
img_dir = 'gen_figures/DE_facescrub_mobile_M100_counter'  # image directory
face_shape = [160, 160]
all_id = 526  # classes need to be evaluated, set it <= num_classes
detector = dlib.get_frontal_face_detector()

# load evaluation model
test_model = Facenet(backbone=eva_backbone, num_classes=num_classes)
test_model.load_state_dict(torch.load(facenet_path, map_location='cpu'), strict=True)
test_model.to(device)
test_model.eval()

# 记录评价指标ACC
acc_number = 0
top5_acc_number = 0
conf_sum = 0
for label in tqdm(range(all_id)):
    label_ = label
    img_path = f'{img_dir}/de_label{label_}_best.jpg'  # image path
    img = Image.open(img_path)
    _, cropped = detect_crop_face(img, detector)
    face_input = facenet_input_preprocessing(cropped, (160, 160)).to(device)

    before_norm, outputs1 = test_model.forward_feature(face_input)
    prediction = test_model.forward_classifier(before_norm)
    conf = F.softmax(prediction, dim=1)
    out = torch.max(conf, dim=1)
    _, rank_ = conf[0].sort(descending=True)
    rank = rank_.to('cpu')

    if label_ == out[1].item():
        acc_number += 1
    if (label_ == np.array(rank[0:5])).sum() > 0:
        top5_acc_number += 1
    conf_sum += conf[0, label_].item()

acc = acc_number / all_id
top5_acc = top5_acc_number / all_id
conf_avg = conf_sum / all_id

print(f'{dataset} attack accuracy: {acc}')
print(f'{dataset} top-5 attack accuracy: {top5_acc}')
print(f'{dataset} attack avg. confidence: {conf_avg}')
