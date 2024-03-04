import sys
sys.path.append('.')
sys.path.append('./src')
sys.path.append('./src/modelinversion')

import os
import argparse
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, CenterCrop, Resize, Compose
from modelinversion.attack.PLGMI_high.gan_trainer import PlgmiGANTrainArgs, PlgmiGANTrainer
from development_config import get_dirs
from modelinversion.attack.PLGMI_high.attacker import PLGMIAttacker, PLGMIAttackConfig
from modelinversion.foldermanager import FolderManager


def get_args():
    parser = argparse.ArgumentParser(description='Stage-1: Train the Pseudo Label-Guided Conditional GAN')
    # Dataset configuration
    parser.add_argument('--data_root', type=str, help='root of public dataset')
    parser.add_argument('--data_name', type=str, help='public dataset')
    parser.add_argument('--target_model', type=str, help='VGG16 | IR152 | FaceNet64')
    parser.add_argument('--target_data_name', type=str, help='VGG16 | IR152 | FaceNet64')
    parser.add_argument('--target_ckpt_path', type=str, help='VGG16 | IR152 | FaceNet64')
    parser.add_argument('--results_root', type=str, default='results',
                        help='Path to results directory. default: results')
    # parser.add_argument('--de', type=str, default='results',
    #                     help='Path to results directory. default: results')
    
    parser.add_argument('--private_data_root', type=str, default='datasets/celeba_private_domain',
                        help='path to private dataset root directory. default: CelebA')
    
    parser.add_argument('--batch_size', '-B', type=int, default=32,
                        help='mini-batch size of training data. default: 64')
    parser.add_argument('--eval_batch_size', '-eB', default=None,
                        help='mini-batch size of evaluation data. default: None')
    # Generator configuration
    parser.add_argument('--gen_num_features', '-gnf', type=int, default=64,
                        help='Number of features of generator (a.k.a. nplanes or ngf). default: 64')
    parser.add_argument('--gen_dim_z', '-gdz', type=int, default=128,
                        help='Dimension of generator input noise. default: 128')
    parser.add_argument('--gen_bottom_width', '-gbw', type=int, default=4,
                        help='Initial size of hidden variable of generator. default: 4')
    parser.add_argument('--gen_distribution', '-gd', type=str, default='normal',
                        help='Input noise distribution: normal (default) or uniform.')
    # Discriminator (Critic) configuration
    parser.add_argument('--dis_num_features', '-dnf', type=int, default=64,
                        help='Number of features of discriminator (a.k.a nplanes or ndf). default: 64')
    # Optimizer settings
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Initial learning rate of Adam. default: 0.0002')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='beta1 (betas[0]) value of Adam. default: 0.0')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='beta2 (betas[1]) value of Adam. default: 0.9')
    # Training setting
    parser.add_argument('--seed', type=int, default=46,
                        help='Random seed. default: 46 (derived from Nogizaka46)')
    parser.add_argument('--max_iteration', '-N', type=int, default=30000,
                        help='Max iteration number of training. default: 30000')
    parser.add_argument('--n_dis', type=int, default=5,
                        help='Number of discriminator updater per generator updater. default: 5')
    parser.add_argument('--num_classes', '-nc', type=int, default=1000,
                        help='Number of classes in training data.  default: 1000')
    parser.add_argument('--loss_type', type=str, default='hinge',
                        help='loss function name. hinge (default) or dcgan.')
    parser.add_argument('--relativistic_loss', '-relloss', default=False, action='store_true',
                        help='Apply relativistic loss or not. default: False')
    parser.add_argument('--calc_FID', default=False, action='store_true',
                        help='If calculate FID score, set this ``True``. default: False')
    # Log and Save interval configuration
    
    parser.add_argument('--no_tensorboard', action='store_true', default=False,
                        help='If you dislike tensorboard, set this ``False``. default: True')
    parser.add_argument('--no_image', action='store_true', default=False,
                        help='If you dislike saving images on tensorboard, set this ``True``. default: False')
    parser.add_argument('--checkpoint_interval', '-ci', type=int, default=1000,
                        help='Interval of saving checkpoints (model and optimizer). default: 1000')
    parser.add_argument('--log_interval', '-li', type=int, default=100,
                        help='Interval of showing losses. default: 100')
    parser.add_argument('--eval_interval', '-ei', type=int, default=1000,
                        help='Interval for evaluation (save images and FID calculation). default: 1000')
    parser.add_argument('--n_eval_batches', '-neb', type=int, default=100,
                        help='Number of mini-batches used in evaluation. default: 100')
    parser.add_argument('--n_fid_images', '-nfi', type=int, default=3000,
                        help='Number of images to calculate FID. default: 5000')
    # Resume training
    parser.add_argument('--args_path', default=None, help='Checkpoint args json path. default: None')
    parser.add_argument('--gen_ckpt_path', '-gcp', default=None,
                        help='Generator and optimizer checkpoint path. default: None')
    parser.add_argument('--dis_ckpt_path', '-dcp', default=None,
                        help='Discriminator and optimizer checkpoint path. default: None')
    # Model Inversion
    parser.add_argument('--alpha', type=float, default=0.2, help='weight of inv loss. default: 0.2')
    parser.add_argument('--inv_loss_type', type=str, default='margin', help='ce | margin | poincare')
    args = parser.parse_args()
    return args

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
if __name__ == '__main__':
    batch_size = 50
    
    more_args = get_args()
    
    dataset_name = more_args.data_name # 'ffhq256'
    target_name = more_args.target_model # 'resnet152'
    target_dataset_name = more_args.target_data_name
    device = 'cuda'
    
    dirs = get_dirs(more_args.results_root)
    
    cache_dir, result_dir, ckpt_dir, dataset_dir = dirs['work_dir'], dirs['result_dir'], dirs['ckpt_dir'], dirs['dataset_dir']

    epoch_num = 50
    
    # trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    train_args = PlgmiGANTrainArgs(
        dataset_name=dataset_name,
        batch_size=batch_size,
        epoch_num=epoch_num,
        device=device,
        target_name=target_name,
        target_dataset_name=target_dataset_name
    )
    
    folder_manager = FolderManager(ckpt_dir, dataset_dir, cache_dir, result_dir, None)
    
    trainer = PlgmiGANTrainer(train_args, folder_manager, args=more_args)
    
    trainer.prepare_training()
    
    del trainer
    
    # eval name support: vgg16, ir152, facenet64, facenet
    eval_name = 'inception_v3'
    # gan target name support: vgg16
    gan_target_name = target_name
    # dataset name support: celeba
    dataset_name = more_args.data_name
    # gan dataset name support: celeba, ffhq, facescrub
    gan_dataset_name = more_args.target_data_name
    
    batch_size = 70
    # target_labels = list(range(512, 544))
    target_labels = list(range(0, 530))
    device = 'cuda'
    
    config = PLGMIAttackConfig(
        target_name=target_name,
        eval_name=eval_name,
        ckpt_dir=ckpt_dir,
        result_dir=result_dir,
        dataset_dir=dataset_dir,
        cache_dir=cache_dir,
        dataset_name=dataset_name,
        device=device,
        gan_target_name=gan_target_name,
        gan_dataset_name=gan_dataset_name,
        gen_num_per_target=50
    )
    
    attacker = PLGMIAttacker(config)
    
    attacker.attack(batch_size, target_labels)
    
    attacker.evaluation(20)