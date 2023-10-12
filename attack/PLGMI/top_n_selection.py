# import os
# import queue
# import shutil
# import torch
# import torch.nn.functional as F
# import torch.utils.data as data
# import torchvision.transforms as transforms
# from PIL import Image
# from argparse import ArgumentParser

# from models import *

# parser = ArgumentParser(description='Reclassify the public dataset with the target model')
# parser.add_argument('--model', default='VGG16', help='VGG16 | IR152 | FaceNet64')
# parser.add_argument('--data_name', type=str, default='celeba', help='celeba | ffhq | facescrub')
# parser.add_argument('--top_n', type=int, help='the n of top-n selection strategy.')
# parser.add_argument('--num_classes', type=int, default=1000)
# parser.add_argument('--save_root', type=str, default='reclassified_public_data')

# args = parser.parse_args()


# class PublicFFHQ(torch.utils.data.Dataset):
#     def __init__(self, root='datasets/ffhq/thumbnails128x128/', transform=None):
#         super(PublicFFHQ, self).__init__()
#         self.root = root
#         self.transform = transform
#         self.images = []
#         self.path = self.root

#         num_classes = len([lists for lists in os.listdir(
#             self.path) if os.path.isdir(os.path.join(self.path, lists))])

#         for idx in range(num_classes):
#             class_path = os.path.join(self.path, str(idx * 1000).zfill(5))
#             for _, _, files in os.walk(class_path):
#                 for img_name in files:
#                     self.images.append(os.path.join(class_path, img_name))

#     def __getitem__(self, index):

#         img_path = self.images[index]
#         # print(img_path)
#         img = Image.open(img_path)
#         if self.transform != None:
#             img = self.transform(img)

#         return img, img_path

#     def __len__(self):
#         return len(self.images)


# class PublicCeleba(torch.utils.data.Dataset):
#     def __init__(self, file_path='data_files/celeba_ganset.txt',
#                  img_root='datasets/celeba/img_align_celeba', transform=None):
#         super(PublicCeleba, self).__init__()
#         self.file_path = file_path
#         self.img_root = img_root
#         self.transform = transform
#         self.images = []

#         name_list, label_list = [], []

#         f = open(self.file_path, "r")
#         for line in f.readlines():
#             img_name = line.strip()
#             self.images.append(os.path.join(self.img_root, img_name))

#     def __getitem__(self, index):

#         img_path = self.images[index]
#         img = Image.open(img_path)
#         if self.transform != None:
#             img = self.transform(img)

#         return img, img_path

#     def __len__(self):
#         return len(self.images)


# class PublicFaceScrub(torch.utils.data.Dataset):
#     def __init__(self, file_path='data_files/facescrub_ganset.txt',
#                  img_root='datasets/facescrub', transform=None):
#         super(PublicFaceScrub, self).__init__()
#         self.file_path = file_path
#         self.img_root = img_root
#         self.transform = transform
#         self.images = []

#         name_list, label_list = [], []

#         f = open(self.file_path, "r")
#         for line in f.readlines():
#             img_name = line.strip()
#             img_path = os.path.join(self.img_root, img_name)
#             try:
#                 if img_path.endswith(".png") or img_path.endswith(".jpg"):
#                     img = Image.open(img_path)
#                     if img.size != (64, 64):
#                         img = img.resize((64, 64), Image.ANTIALIAS)
#                     img = img.convert('RGB')
#                     self.images.append((img, img_path))
#             except:
#                 continue

#     def __getitem__(self, index):

#         img, img_path = self.images[index]
#         if self.transform != None:
#             img = self.transform(img)

#         return img, img_path

#     def __len__(self):
#         return len(self.images)


# def top_n_selection(args, T, data_loader):
#     """
#     Top-n selection strategy.
#     :param args: top-n, save_path
#     :param T: target model
#     :param data_loader: dataloader of
#     :return:
#     """
#     print("=> start inference ...")
#     all_images_prob = None
#     all_images_path = None
#     # get the predict confidence of each image in the public data
#     with torch.no_grad():
#         for i, (images, img_path) in enumerate(data_loader):
#             bs = images.shape[0]
#             images = images.cuda()
#             logits = T(images)[-1]
#             prob = F.softmax(logits, dim=1)  # (bs, 1000)
#             prob = prob.cpu()
#             if i == 0:
#                 all_images_prob = prob
#                 all_images_path = img_path
#             else:
#                 all_images_prob = torch.cat([all_images_prob, prob], dim=0)
#                 all_images_path = all_images_path + img_path

#     print("=> start reclassify ...")
#     save_path = os.path.join(args.save_root, args.data_name, args.model + "_top" + str(args.top_n))
#     print(" top_n: ", args.top_n)
#     print(" save_path: ", save_path)
#     # top-n selection
#     for class_idx in range(args.num_classes):
#         bs = all_images_prob.shape[0]
#         ccc = 0
#         # maintain a priority queue
#         q = queue.PriorityQueue()
#         class_idx_prob = all_images_prob[:, class_idx]

#         for j in range(bs):
#             current_value = float(class_idx_prob[j])
#             image_path = all_images_path[j]
#             # Maintain a priority queue with confidence as the priority
#             if q.qsize() < args.top_n:
#                 q.put([current_value, image_path])
#             else:
#                 current_min = q.get()
#                 if current_value < current_min[0]:
#                     q.put(current_min)
#                 else:
#                     q.put([current_value, image_path])
#         # reclassify and move the images
#         for m in range(q.qsize()):
#             q_value = q.get()
#             q_prob = round(q_value[0], 4)
#             q_image_path = q_value[1]

#             ori_save_path = os.path.join(save_path, str(class_idx))
#             if not os.path.exists(ori_save_path):
#                 os.makedirs(ori_save_path)

#             new_image_path = os.path.join(ori_save_path, str(ccc) + '_' + str(q_prob) + '.png')

#             shutil.copy(q_image_path, new_image_path)
#             ccc += 1


# print(args)
# print("=> load target model ...")

# model_name_T = args.model
# if model_name_T.startswith("VGG16"):
#     T = VGG16(1000)
#     path_T = 'checkpoints/target_model/VGG16_88.26.tar'
#     print(path_T)
# elif model_name_T.startswith('IR152'):
#     T = IR152(1000)
#     path_T = 'checkpoints/target_model/IR152_91.16.tar'
# elif model_name_T == "FaceNet64":
#     T = FaceNet64(1000)
#     path_T = 'checkpoints/target_model/aceNet64_88.50.tar'

# T = torch.nn.DataParallel(T).cuda()
# ckp_T = torch.load(path_T)
# T.load_state_dict(ckp_T['state_dict'], strict=False)
# T.eval()

# print("=> load public dataset ...")
# if args.data_name == 'celeba':
#     re_size = 64
#     crop_size = 108
#     offset_height = (218 - crop_size) // 2
#     offset_width = (178 - crop_size) // 2
#     crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
#     celeba_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Lambda(crop),
#         transforms.ToPILImage(),
#         transforms.Resize((re_size, re_size)),
#         transforms.ToTensor()
#     ])
#     data_set = PublicCeleba(file_path='data_files/celeba_ganset.txt',
#                             img_root='datasets/celeba/img_align_celeba',
#                             transform=celeba_transform)
#     data_loader = data.DataLoader(data_set, batch_size=350)
# elif args.data_name == 'ffhq':
#     re_size = 64
#     crop_size = 88
#     offset_height = (128 - crop_size) // 2
#     offset_width = (128 - crop_size) // 2
#     crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
#     ffhq_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Lambda(crop),
#         transforms.ToPILImage(),
#         transforms.Resize((re_size, re_size)),
#         transforms.ToTensor()
#     ])
#     data_set = PublicFFHQ(root='datasets/ffhq/thumbnails128x128/', transform=ffhq_transform)
#     data_loader = data.DataLoader(data_set, batch_size=350)
# elif args.data_name == 'facescrub':
#     crop_size = 54
#     offset_height = (64 - crop_size) // 2
#     offset_width = (64 - crop_size) // 2
#     re_size = 64
#     crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

#     faceScrub_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Lambda(crop),
#         transforms.ToPILImage(),
#         transforms.Resize((re_size, re_size)),
#         transforms.ToTensor()
#     ])
#     data_set = PublicFaceScrub(file_path='data_files/facescrub_ganset.txt',
#                                img_root='datasets/facescrub', transform=faceScrub_transform)
#     data_loader = data.DataLoader(data_set, batch_size=350)

# top_n_selection(args, T, data_loader)
