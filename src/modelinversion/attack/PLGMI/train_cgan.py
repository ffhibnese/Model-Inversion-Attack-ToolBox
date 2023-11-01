import argparse
import datetime
import json
import kornia
import os
import shutil
import torch
import torch.nn.functional as F
import torchvision

# from . import evaluation
from . import losses as L
from . import utils
from .dataset import FaceDataset, InfiniteSamplerWrapper, sample_from_data, sample_from_gen
from .models import inception
from ...models import VGG16, FaceNet, IR152, FaceNet64
from .models.discriminators.snresnet64 import SNResNetProjectionDiscriminator
from .models.generators.resnet64 import ResNetGenerator


def prepare_results_dir(args):
    """Makedir, init tensorboard if required, save args."""
    root = os.path.join(args.results_root,
                        args.data_name, args.target_model)
    os.makedirs(root, exist_ok=True)
    if not args.no_tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(root)
    else:
        writer = None

    train_image_root = os.path.join(root, "preview", "train")
    eval_image_root = os.path.join(root, "preview", "eval")
    os.makedirs(train_image_root, exist_ok=True)
    os.makedirs(eval_image_root, exist_ok=True)

    args.results_root = root
    args.train_image_root = train_image_root
    args.eval_image_root = eval_image_root

    if args.num_classes > args.n_eval_batches:
        args.n_eval_batches = args.num_classes
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size // 4

    if args.calc_FID:
        args.n_fid_batches = args.n_fid_images // args.batch_size

    with open(os.path.join(root, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print(json.dumps(args.__dict__, indent=2))
    return args, writer


def get_args():
    parser = argparse.ArgumentParser(description='Stage-1: Train the Pseudo Label-Guided Conditional GAN')
    # Dataset configuration
    parser.add_argument('--data_root', type=str, help='path to dataset root directory.')
    parser.add_argument('--data_name', type=str, help='celeba | ffhq | facescrub')
    parser.add_argument('--target_model', type=str, help='VGG16 | IR152 | FaceNet64')
    parser.add_argument('--private_data_root', type=str, default='datasets/celeba_private_domain',
                        help='path to private dataset root directory. default: CelebA')
    parser.add_argument('--batch_size', '-B', type=int, default=64,
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
    parser.add_argument('--results_root', type=str, default='results',
                        help='Path to results directory. default: results')
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


def main():
    args = get_args()

    # load target model
    print("Target Model:", args.target_model)
    if args.target_model.startswith("VGG16"):
        target_model = VGG16(args.num_classes)
        target_model_path = 'checkpoints/target_model/VGG16_88.26.tar'
    elif args.target_model.startswith('IR152'):
        target_model = IR152(args.num_classes)
        target_model_path = 'checkpoints/target_model/IR152_91.16.tar'
    elif args.target_model == "FaceNet64":
        target_model = FaceNet64(args.num_classes)
        target_model_path = 'checkpoints/target_model/FaceNet64_88.50.tar'

    target_model = torch.nn.DataParallel(target_model).cuda()
    target_model.load_state_dict(torch.load(target_model_path)['state_dict'], strict=False)
    target_model.eval()

    # load evaluate model
    evaluate_model = FaceNet(args.num_classes)
    evaluate_model_path = 'checkpoints/evaluate_model/FaceNet_95.88.tar'
    evaluate_model = torch.nn.DataParallel(evaluate_model).cuda()
    evaluate_model.load_state_dict(torch.load(evaluate_model_path)['state_dict'], strict=False)
    evaluate_model.eval()

    # CUDA setting
    if not torch.cuda.is_available():
        raise ValueError("Should buy GPU!")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True

    def _noise_adder(img):
        return torch.empty_like(img, dtype=img.dtype).uniform_(0.0, 1 / 256.0) + img

    # dataset crop setting
    if args.data_name == 'celeba':
        re_size = 64
        crop_size = 108
        offset_height = (218 - crop_size) // 2
        offset_width = (178 - crop_size) // 2
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
    elif args.data_name == 'ffhq':
        crop_size = 88
        offset_height = (128 - crop_size) // 2
        offset_width = (128 - crop_size) // 2
        re_size = 64
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
    elif args.data_name == 'facescrub':
        re_size = 64
        crop_size = 64
        offset_height = (64 - crop_size) // 2
        offset_width = (64 - crop_size) // 2
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
    else:
        print("Wrong Dataname!")

    # load public dataset
    my_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(crop),
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((re_size, re_size)),
        torchvision.transforms.ToTensor(),
        _noise_adder
    ])
    train_dataset = FaceDataset(args=args, root=args.data_root, transform=my_transform)
    train_loader = iter(torch.utils.data.DataLoader(
        train_dataset, args.batch_size,
        sampler=InfiniteSamplerWrapper(train_dataset),
    )
    )
    # calculate the FID of generated images
    if args.calc_FID:
        eval_dataset = torchvision.datasets.ImageFolder(
            args.private_data_root,
            torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
        )
        eval_loader = iter(torch.utils.data.DataLoader(
            eval_dataset, args.batch_size,
            sampler=InfiniteSamplerWrapper(eval_dataset)
        )
        )

    else:
        eval_loader = None

    print(' prepared datasets...')
    print(' Number of training images: {}'.format(len(train_dataset)))
    # Prepare directories.
    args, writer = prepare_results_dir(args)
    # initialize generator and discriminator.
    _n_cls = args.num_classes
    gen = ResNetGenerator(
        args.gen_num_features, args.gen_dim_z, args.gen_bottom_width,
        activation=F.relu, num_classes=_n_cls, distribution=args.gen_distribution
    ).to(device)
    dis = SNResNetProjectionDiscriminator(args.dis_num_features, _n_cls, F.relu).to(device)
    inception_model = inception.InceptionV3().to(device) if args.calc_FID else None  # Calc FID need
    # load optimizer
    opt_gen = torch.optim.Adam(gen.parameters(), args.lr, (args.beta1, args.beta2))
    opt_dis = torch.optim.Adam(dis.parameters(), args.lr, (args.beta1, args.beta2))
    # get adversarial loss
    gen_criterion = L.GenLoss(args.loss_type, args.relativistic_loss)
    dis_criterion = L.DisLoss(args.loss_type, args.relativistic_loss)
    print(' Initialized models...\n')

    if args.args_path is not None:
        print(' Load weights...\n')
        prev_args, gen, opt_gen, dis, opt_dis = utils.resume_from_args(
            args.args_path, args.gen_ckpt_path, args.dis_ckpt_path
        )
    # data augmentation module in stage-1 for the generated images
    aug_list = kornia.augmentation.container.ImageSequential(
        kornia.augmentation.RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        kornia.augmentation.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
        kornia.augmentation.RandomHorizontalFlip(),
        kornia.augmentation.RandomRotation(5),
    )

    # Training loop
    for n_iter in range(1, args.max_iteration + 1):
        # ==================== Beginning of 1 iteration. ====================
        _l_g = .0
        cumulative_inv_loss = 0.
        cumulative_loss_dis = .0

        cumulative_target_acc = .0
        target_correct = 0
        count = 0
        for i in range(args.n_dis):  # args.ndis=5, Gen update 1 time, Dis update ndis times.
            if i == 0:
                fake, pseudo_y, _ = sample_from_gen(args, device, args.num_classes, gen)
                dis_fake = dis(fake, pseudo_y)
                # random transformation on the generated images
                fake_aug = aug_list(fake)
                # calc the L_inv
                if args.inv_loss_type == 'ce':
                    inv_loss = L.cross_entropy_loss(target_model(fake_aug)[-1], pseudo_y)
                elif args.inv_loss_type == 'margin':
                    inv_loss = L.max_margin_loss(target_model(fake_aug)[-1], pseudo_y)
                elif args.inv_loss_type == 'poincare':
                    inv_loss = L.poincare_loss(target_model(fake_aug)[-1], pseudo_y)
                # not used
                if args.relativistic_loss:
                    real, y = sample_from_data(args, device, train_loader)
                    dis_real = dis(real, y)
                else:
                    dis_real = None
                # calc the loss of G
                loss_gen = gen_criterion(dis_fake, dis_real)
                loss_all = loss_gen + inv_loss * args.alpha
                # update the G
                gen.zero_grad()
                loss_all.backward()
                opt_gen.step()
                _l_g += loss_gen.item()

                cumulative_inv_loss += inv_loss.item()

                if n_iter % 10 == 0 and writer is not None:
                    writer.add_scalar('gen', _l_g, n_iter)
                    writer.add_scalar('inv', cumulative_inv_loss, n_iter)
            # generate fake images
            fake, pseudo_y, _ = sample_from_gen(args, device, args.num_classes, gen)
            # sample the real images
            real, y = sample_from_data(args, device, train_loader)
            # calc the loss of D
            dis_fake, dis_real = dis(fake, pseudo_y), dis(real, y)
            loss_dis = dis_criterion(dis_fake, dis_real)
            # update D
            dis.zero_grad()
            loss_dis.backward()
            opt_dis.step()

            cumulative_loss_dis += loss_dis.item()

            with torch.no_grad():
                count += fake.shape[0]
                T_logits = target_model(fake)[-1]
                T_preds = T_logits.max(1, keepdim=True)[1]
                target_correct += T_preds.eq(pseudo_y.view_as(T_preds)).sum().item()
                cumulative_target_acc += round(target_correct / count, 4)

            if n_iter % 10 == 0 and i == args.n_dis - 1 and writer is not None:
                cumulative_loss_dis /= args.n_dis
                cumulative_target_acc /= args.n_dis
                writer.add_scalar('dis', cumulative_loss_dis, n_iter)
                writer.add_scalar('target acc', cumulative_target_acc, n_iter)
        # ==================== End of 1 iteration. ====================

        if n_iter % args.log_interval == 0:
            print(
                'iteration: {:07d}/{:07d}, loss gen: {:05f}, loss dis {:05f}, inv loss {:05f}, target acc {:04f}'.format(
                    n_iter, args.max_iteration, _l_g, cumulative_loss_dis, cumulative_inv_loss,
                    cumulative_target_acc, ))
            if not args.no_image:
                writer.add_image(
                    'fake', torchvision.utils.make_grid(
                        fake, nrow=4, normalize=True, scale_each=True))
                writer.add_image(
                    'real', torchvision.utils.make_grid(
                        real, nrow=4, normalize=True, scale_each=True))
            # Save previews
            utils.save_images(
                n_iter, n_iter // args.checkpoint_interval, args.results_root,
                args.train_image_root, fake, real
            )
        if n_iter % args.checkpoint_interval == 0:
            # Save checkpoints!
            utils.save_checkpoints(
                args, n_iter, n_iter // args.checkpoint_interval,
                gen, opt_gen, dis, opt_dis
            )
        # if n_iter % args.eval_interval == 0:
        #     # Once these criterion are prepared, val_loader will be used.
        #     fid_score = evaluation.evaluate(
        #         args, n_iter, gen, device, inception_model, eval_loader
        #     )
        #     print('[Eval] iteration: {:07d}/{:07d}, FID: {:07f}'.format(
        #         n_iter, args.max_iteration, fid_score))
        #     if writer is not None:
        #         writer.add_scalar("FID", fid_score, n_iter)
        #         # Project embedding weights if exists.
        #         embedding_layer = getattr(dis, 'l_y', None)
        #         if embedding_layer is not None:
        #             writer.add_embedding(
        #                 embedding_layer.weight.data,
        #                 list(range(args.num_classes)),
        #                 global_step=n_iter
        #             )


if __name__ == '__main__':
    main()
