import torch
from networks.facenet import Facenet
from train_facenet import get_num_classes
from utils.detect_crop_face import facenet_input_preprocessing, detect_crop_face
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm
import dlib
import numpy as np

device        = 'cuda'
img_size      = 256
dataset       = 'facescrub'
t_model_bone  = 'mobile_net'
eva_backbone  = 'inception_resnetv1'
facenet_path  = ''  # path to evaluation model
num_classes   = get_num_classes('')
img_dir       = ''
face_shape    = [160, 160]
all_id        = 526
detector      = dlib.get_frontal_face_detector()

# load evaluation model
test_model = Facenet(backbone=eva_backbone, num_classes=num_classes)
test_model.load_state_dict(torch.load(facenet_path, map_location='cpu'), strict=True)
test_model.to(device)
test_model.eval()

acc_number = 0
top5_acc_number = 0
conf_sum   = 0
for label in tqdm(range(all_id)):
    label_ = label
    img_path      = f'{img_dir}/label{label_}.jpg'
    img = Image.open(img_path)
    _, cropped = detect_crop_face(img, detector)
    face_input = facenet_input_preprocessing(cropped,(160,160)).to(device)

    before_norm, outputs1 = test_model.forward_feature(face_input)
    prediction = test_model.forward_classifier(before_norm)
    conf = F.softmax(prediction, dim=1)
    out  = torch.max(conf, dim=1)
    _, rank_ = conf[0].sort(descending=True)
    rank = rank_.to('cpu')

    if label_ == out[1].item():
        acc_number += 1
    if (label_ == np.array(rank[0:5])).sum() > 0:
        top5_acc_number += 1
    conf_sum += conf[0,label_].item()

acc      = acc_number / all_id
top5_acc = top5_acc_number / all_id
conf_avg = conf_sum / all_id

print(f'{dataset} attack accuracy: {acc}')
print(f'{dataset} top-5 attack accuracy: {top5_acc}')
print(f'{dataset} attack avg. confidence: {conf_avg}')
