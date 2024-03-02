import sys
sys.path.append('/data/yuhongyao/Model_Inversion_Attack_ToolBox/test')

import torch
import torchvision.transforms as T
import traceback
from facenet_pytorch import InceptionResnetV1
from torch.utils.data import TensorDataset

from celeba import *
from facescrub import *
from stanford_dogs import *
from custom_subset import *
from fid_score import FID_Score
from prcd import PRCD
from classifier import Classifier
from distance_metrics import DistanceEvaluation
from classifier import Classifier

class evaluator():
    def __init__(self, evaluation_model_dist, target_dataset, batch_size, gpu_devices, device='cuda'):
        self.device = device
        self.devices = gpu_devices
        # set transformations
        crop_size = 800
        target_transform = T.Compose([
            T.ToTensor(),
            T.Resize((299, 299), antialias=True),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        facenet = InceptionResnetV1(pretrained='vggface2')
        facenet = torch.nn.DataParallel(
            facenet, device_ids=gpu_devices)
        facenet.to(device)
        facenet.eval()
        evaluation_model_dist.model.fc = torch.nn.Sequential()
        evaluation_model_dist = torch.nn.DataParallel(evaluation_model_dist,
                                                      device_ids=gpu_devices)
        evaluation_model_dist.to(device)
        evaluation_model_dist.eval()
        self.evaluate_inception = DistanceEvaluation(
            evaluation_model_dist,
            299,
            crop_size,
            target_dataset, 42)
        self.evaluate_facenet = DistanceEvaluation(facenet, 160,
                                                   crop_size,
                                                   target_dataset, 42)

        self.full_dataset = create_target_dataset(target_dataset,
                                                  target_transform)
        self.fid_evaluation = FID_Score(device=device,
                                        crop_size=crop_size,
                                        batch_size=batch_size * 3,
                                        dims=2048,
                                        num_workers=8,
                                        gpu_devices=gpu_devices)
        self.prdc = PRCD(device=device,
                         crop_size=crop_size,
                         batch_size=batch_size * 3,
                         dims=2048,
                         num_workers=8,
                         gpu_devices=gpu_devices)

    def compute(self, imgs, targets, target_dataset, batch_size):
        ####################################
        #    FID Score and GAN Metrics     #
        ####################################
        try:
            # create datasets
            attack_dataset = TensorDataset(imgs, targets)
            attack_dataset.targets = targets

            training_dataset = ClassSubset(
                self.full_dataset,
                target_classes=torch.unique(targets).cpu().tolist())

            # compute FID score
            self.fid_evaluation.set(training_dataset, attack_dataset)
            self.fid_evaluation.compute_fid()

            # compute precision, recall, density, coverage
            self.prdc.set(training_dataset, attack_dataset)
            self.prdc.compute_metric(int(targets[0]), k=3)

        except Exception:
            print(traceback.format_exc())

        ####################################
        #         Feature Distance         #
        ####################################
        try:

            # Compute average feature distance on Inception-v3
            self.evaluate_inception.compute_dist(
                imgs,
                targets,
                batch_size=batch_size * 5)

            # Compute feature distance only for facial images
            if target_dataset in [
                    'facescrub', 'celeba_identities', 'celeba_attributes'
            ]:
                self.evaluate_facenet.compute_dist(
                    imgs,
                    targets,
                    batch_size=batch_size * 8)
        except Exception:
            print(traceback.format_exc())
    
    def get_fid(self):
        fid_score = self.fid_evaluation.get_fid()
        return fid_score
    
    def get_prdc(self):
        precision, recall, density, coverage = self.prdc.get_prcd()
        return precision, recall, density, coverage
    
    def get_incv_dist(self):
        avg_dist_inception, mean_distances_list = self.evaluate_inception.get_eval_dist()
        return avg_dist_inception, mean_distances_list

    def get_face_dist(self):
        avg_dist_facenet, mean_distances_list = self.evaluate_facenet.get_eval_dist()
        return avg_dist_facenet, mean_distances_list


def create_target_dataset(dataset_name, transform):
    if dataset_name.lower() == 'facescrub':
        return FaceScrub(group='all',
                         train=True,
                         transform=transform)
    elif dataset_name.lower() == 'celeba_identities':
        return CelebA1000(train=True, transform=transform)
    elif 'stanford_dogs' in dataset_name.lower():
        return StanfordDogs(train=True, cropped=True, transform=transform)
    else:
        print(f'{dataset_name} is no valid dataset.')


def create_evaluation_model(num_classes, architecture, path):
    evaluation_model = Classifier(num_classes=num_classes,
                                  architecture=architecture)
    evaluation_model.load_state_dict(torch.load(path))
    return evaluation_model

from tqdm import tqdm
def main():
    root_path = '/data/yuhongyao/Model_Inversion_Attack_ToolBox/results/lomma_kedmi_high_metfaces_facescrub/facescrub_resnet18_metfaces_vgg16/all_imgs'
    all_tensors = []
    all_labels = []
    import torchvision
    ds = torchvision.datasets.ImageFolder(root_path, transform=torchvision.transforms.ToTensor())
    all_infos = [[img, label] for img, label in tqdm(ds)]
    all_labels = [label for _, label in tqdm(all_infos)]
    all_tensors = [t for t, _ in tqdm(all_infos)]

    # for label in tqdm(os.listdir(root_path)):
    #     label_dir = os.path.join(root_path, label)
    #     label_tensors = []
    #     for filename in os.listdir(label_dir):
    #         if filename.endswith('.pt'):
    #             filepath = os.path.join(label_dir, filename)
    #             tensor = torch.load(filepath, map_location='cpu')
    #             # print(tensor.shape)
    #             label_tensors.append(tensor)
    #     # break
    #     label_tensors = torch.stack(label_tensors, dim=0)
    #     label_targets = torch.ones(len(label_tensors), dtype=torch.long) * int(label)
    #     all_tensors.append(label_tensors)
    #     all_labels.append(label_targets)
    all_tensors = torch.stack(all_tensors, dim=0)
    all_labels = torch.LongTensor(all_labels)
    print(all_tensors.shape, all_labels.shape)
    
    eval_model = Classifier(num_classes=530, architecture='inception_v3')
    eval_model.load_state_dict(torch.load('../../../intermediate-MIA/intermediate-MIA/pretrained/inception_v3_facescrub.pt'))
    eva = evaluator(eval_model, 'facescrub', 20, gpu_devices=[0])
    print('start evaluate')
    eva.compute(all_tensors, all_labels, 'facescrub', 20)
    
    print('fid', eva.get_fid())
    print('eval dist', eva.get_incv_dist()[0])
    print('face dist', eva.get_face_dist()[0])
    print('prcd', eva.get_prdc())
    
if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    
    main()