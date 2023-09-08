import numpy as np
import torch, random, sys, json, time, dataloader
import torch.nn as nn
from datetime import datetime
from torch.utils.data import sampler
import torchvision.utils as tvls

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        if not '...' in data:
            self.file.write(data)
        self.stdout.write(data)
        self.flush()
    def flush(self):
        self.file.flush()

def get_center_mask(img_size=64):
    
    mask = torch.zeros(img_size, img_size).cuda()
    scale = 0.25
    l = int(img_size * scale)
    u = int(img_size * (1.0 - scale))
    mask[l:u, l:u] = 1
    return mask

def get_input_mask(img_size, bs):
    typ = random.randint(0, 1)
    mask = torch.zeros(img_size, img_size).cuda().float()
    
    if typ == 0:
        scale = 0.15
        l = int(img_size * scale)
        u = int(img_size * (1.0 - scale))
        mask[l:u, l:u] = 1
    elif typ == 1:
        u, d = 10, 52
        l, r = 25, 40
        mask[l:r, u:d] = 1
        u, d = 26, 38
        l, r = 40, 63
        mask[l:r, u:d] = 1
        
    mask = mask.repeat(bs, 3, 1, 1)
    return mask

def load_state_dict(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param.data)

def load_pretrain(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name.startswith("module.fc_layer"):
            continue
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param.data)

def load_params(self, model):
    own_state = self.state_dict()
    for name, param in model.named_parameters():
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param.data)

def load_json(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data

def print_params(info, params, dataset=None):
    print('-----------------------------------------------------------------')
    if dataset is not None:
        print("Dataset: %s" % dataset)
        print("Running time: %s" % datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    for i, (key, value) in enumerate(info.items()):
        if i >=3: 
            print("%s: %s" % (key, str(value)))
    for i, (key, value) in enumerate(params.items()):
        print("%s: %s" % (key, str(value)))
    print('-----------------------------------------------------------------')

def init_dataloader(args, file_path, batch_size=64, mode="gan"):
    tf = time.time()
    if args['dataset']['name'] == "celeba":
        data_set = dataloader.ImageFolder(args, file_path, mode)
    elif args['dataset']['name'] == "MNIST":
        data_set = dataloader.GrayFolder(args, file_path, mode)

    data_loader = torch.utils.data.DataLoader(data_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=4,
                                              pin_memory=True)
    
    interval = time.time() - tf
    print('Initializing data loader took {:.2f}'.format(interval))
    return data_loader

def save_tensor_images(images, filename, nrow=None, normalize=True):
    if not nrow:
        tvls.save_image(images, filename, normalize = normalize, padding=0)
    else:
        tvls.save_image(images, filename, normalize = normalize, nrow=nrow, padding=0)

def load_my_state_dict(self, state_dict):
    own_state = self.state_dict()
    #print(state_dict)
    for name, param in state_dict.items():
        if name not in own_state:
            print(name)
            continue
        #print(param.data.shape)
        own_state[name].copy_(param.data)

