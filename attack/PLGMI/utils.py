import json
import numpy
import os
import shutil
import torch
import torchvision


class Dict2Args(object):
    """Dict-argparse object converter."""

    def __init__(self, dict_args):
        for key, value in dict_args.items():
            setattr(self, key, value)


def generate_images(gen, device, batch_size=64, dim_z=128, distribution=None,
                    num_classes=None, class_id=None):
    """Generate images.

    Priority: num_classes > class_id.

    Args:
        gen (nn.Module): generator.
        device (torch.device)
        batch_size (int)
        dim_z (int)
        distribution (str)
        num_classes (int, optional)
        class_id (int, optional)

    Returns:
        torch.tensor

    """

    z = sample_z(batch_size, dim_z, device, distribution)
    if num_classes is None and class_id is None:
        y = None
    elif num_classes is not None:
        y = sample_pseudo_labels(num_classes, batch_size, device)
    elif class_id is not None:
        y = torch.tensor([class_id] * batch_size, dtype=torch.long).to(device)
    else:
        y = None
    with torch.no_grad():
        fake = gen(z, y)

    return fake


def sample_z(batch_size, dim_z, device, distribution=None):
    """Sample random noises.

    Args:
        batch_size (int)
        dim_z (int)
        device (torch.device)
        distribution (str, optional): default is normal

    Returns:
        torch.FloatTensor or torch.cuda.FloatTensor

    """

    if distribution is None:
        distribution = 'normal'
    if distribution == 'normal':
        return torch.empty(batch_size, dim_z, dtype=torch.float32, device=device).normal_()
    else:
        return torch.empty(batch_size, dim_z, dtype=torch.float32, device=device).uniform_()


def sample_pseudo_labels(num_classes, batch_size, device):
    """Sample pseudo-labels.

    Args:
        num_classes (int): number of classes in the dataset.
        batch_size (int): size of mini-batch.
        device (torch.Device): For compatibility.

    Returns:
        ~torch.LongTensor or torch.cuda.LongTensor.

    """

    pseudo_labels = torch.from_numpy(
        numpy.random.randint(low=0, high=num_classes, size=(batch_size))
    )
    pseudo_labels = pseudo_labels.type(torch.long).to(device)
    return pseudo_labels


def save_images(n_iter, count, root, train_image_root, fake, real):
    """Save images (torch.tensor).

    Args:
        root (str)
        train_image_root (root)
        fake (torch.tensor)
        real (torch.tensor)

    """

    fake_path = os.path.join(
        train_image_root,
        'fake_{}_iter_{:07d}.png'.format(count, n_iter)
    )
    real_path = os.path.join(
        train_image_root,
        'real_{}_iter_{:07d}.png'.format(count, n_iter)
    )
    torchvision.utils.save_image(
        fake, fake_path, nrow=4, normalize=True, scale_each=True
    )
    shutil.copy(fake_path, os.path.join(root, 'fake_latest.png'))
    torchvision.utils.save_image(
        real, real_path, nrow=4, normalize=True, scale_each=True
    )
    shutil.copy(real_path, os.path.join(root, 'real_latest.png'))


def save_checkpoints(args, n_iter, count, gen, opt_gen, dis, opt_dis):
    """Save checkpoints.

    Args:
        args (argparse object)
        n_iter (int)
        gen (nn.Module)
        opt_gen (torch.optim)
        dis (nn.Module)
        opt_dis (torch.optim)

    """

    count = n_iter // args.checkpoint_interval
    gen_dst = os.path.join(
        args.results_root,
        'gen_{}_iter_{:07d}.pth.tar'.format(count, n_iter)
    )
    torch.save({
        'model': gen.state_dict(), 'opt': opt_gen.state_dict(),
    }, gen_dst)
    shutil.copy(gen_dst, os.path.join(args.results_root, 'gen_latest.pth.tar'))
    dis_dst = os.path.join(
        args.results_root,
        'dis_{}_iter_{:07d}.pth.tar'.format(count, n_iter)
    )
    torch.save({
        'model': dis.state_dict(), 'opt': opt_dis.state_dict(),
    }, dis_dst)
    shutil.copy(dis_dst, os.path.join(args.results_root, 'dis_latest.pth.tar'))


def resume_from_args(args_path, gen_ckpt_path, dis_ckpt_path):
    """Load generator & discriminator with their optimizers from args.json.

    Args:
        args_path (str): Path to args.json
        gen_ckpt_path (str): Path to generator checkpoint or relative path
                             from args['results_root']
        dis_ckpt_path (str): Path to discriminator checkpoint or relative path
                             from args['results_root']

    Returns:
        gen, opt_dis
        dis, opt_dis

    """

    from models.generators import resnet64
    from models.discriminators import snresnet64

    with open(args_path) as f:
        args = json.load(f)
    conditional = args['cGAN']
    num_classes = args['num_classes'] if conditional else 0
    # Initialize generator
    gen = resnet64.ResNetGenerator(
        args['gen_num_features'], args['gen_dim_z'], args['gen_bottom_width'],
        num_classes=num_classes, distribution=args['gen_distribution']
    )
    opt_gen = torch.optim.Adam(
        gen.parameters(), args['lr'], (args['beta1'], args['beta2'])
    )
    # Initialize discriminator
    if args['dis_arch_concat']:
        dis = snresnet64.SNResNetConcatDiscriminator(
            args['dis_num_features'], num_classes, dim_emb=args['dis_emb']
        )
    else:
        dis = snresnet64.SNResNetProjectionDiscriminator(
            args['dis_num_features'], num_classes
        )
    opt_dis = torch.optim.Adam(
        dis.parameters(), args['lr'], (args['beta1'], args['beta2'])
    )
    if not os.path.exists(gen_ckpt_path):
        gen_ckpt_path = os.path.join(args['results_root'], gen_ckpt_path)
    gen, opt_gen = load_model_optim(gen_ckpt_path, gen, opt_gen)
    if not os.path.exists(dis_ckpt_path):
        dis_ckpt_path = os.path.join(args['results_root'], dis_ckpt_path)
    dis, opt_dis = load_model_optim(dis_ckpt_path, dis, opt_dis)
    return Dict2Args(args), gen, opt_gen, dis, opt_dis


def load_model_optim(checkpoint_path, model=None, optim=None):
    """Load trained weight.

    Args:
        checkpoint_path (str)
        model (nn.Module)
        optim (torch.optim)

    Returns:
        model
        optim

    """

    ckpt = torch.load(checkpoint_path)
    if model is not None:
        model.load_state_dict(ckpt['model'])
    if optim is not None:
        optim.load_state_dict(ckpt['opt'])
    return model, optim


def load_model(checkpoint_path, model):
    """Load trained weight.

    Args:
        checkpoint_path (str)
        model (nn.Module)

    Returns:
        model

    """

    return load_model_optim(checkpoint_path, model, None)[0]


def load_optim(checkpoint_path, optim):
    """Load optimizer from checkpoint.

    Args:
        checkpoint_path (str)
        optim (torch.optim)

    Returns:
        optim

    """

    return load_model_optim(checkpoint_path, None, optim)[1]


def save_tensor_images(images, filename, nrow=None, normalize=True):
    if not nrow:
        torchvision.utils.save_image(images, filename, normalize=normalize, padding=0)
    else:
        torchvision.utils.save_image(images, filename, normalize=normalize, nrow=nrow, padding=0)
