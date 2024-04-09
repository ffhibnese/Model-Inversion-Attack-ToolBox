import os
import numpy as np
import sys
from tqdm import tqdm
from copy import deepcopy

# InsightFace model
INSIGHTFACE_ROOT = '../InsightFace_Pytorch'
sys.path.append(INSIGHTFACE_ROOT)
from config import get_config

# from Learner import face_learner
from model import Backbone

# from data.data_pipe import get_val_pair
import torchvision.utils as vutils
import torch

# import bcolz
from torchvision import transforms as trans
from celeba import get_celeba_dataset

# Finetuning
import utils
import torch.nn as nn
from DiffAugment_pytorch import DiffAugment
import torch.nn.functional as F
from utils import mkdir
from csv_logger import CSVLogger, plot_csv


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    return torch.pow(
        x.unsqueeze(1).expand(n, m, d) - y.unsqueeze(0).expand(n, m, d), 2
    ).sum(2)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def de_preprocess(tensor):
    return tensor * 0.5 + 0.5


hflip = trans.Compose(
    [
        de_preprocess,
        trans.ToPILImage(),
        trans.functional.hflip,
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)
    return hfliped_imgs


rs112 = trans.Compose(
    [
        de_preprocess,
        trans.ToPILImage(),
        trans.Resize(112),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


def resize112_batch(imgs_tensor):
    device = imgs_tensor.device
    resized_imgs = torch.zeros(len(imgs_tensor), 3, 112, 112)
    for i, img_ten in enumerate(imgs_tensor):
        resized_imgs[i] = rs112(img_ten.cpu())
    return resized_imgs.to(device)


def padto112_batch(imgs_tensor):
    assert imgs_tensor.shape[-1] == 64
    padded_imgs = torch.zeros(len(imgs_tensor), 3, 112, 112).to(imgs_tensor.device)
    padded_imgs[:, :, 24:-24, 24:-24] = imgs_tensor
    return padded_imgs


def trim_batch(imgs_tensor):
    assert imgs_tensor.shape[-1] == 112
    mask = torch.zeros_like(imgs_tensor)
    mask[:, :, 24:-24, 24:-24] += 1
    return imgs_tensor * mask


def embedding_dist(embeddings1, embeddings2):
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    return dist


def insightface_fpass(
    x, device, model, embedding_size, tta=True, batch_size=1000, pad=False
):
    embeddings = torch.from_numpy(np.zeros([len(x), embedding_size])).to(device)
    with torch.no_grad():
        for idx in tqdm(range(0, len(x), batch_size), desc=f'insightface_fpass '):
            batch = torch.tensor(x[idx : idx + batch_size])
            if pad:
                batch = padto112_batch(batch)
            else:
                batch = resize112_batch(batch)
            if tta:
                fliped = hflip_batch(batch)
                emb_batch = model(batch.to(device)) + model(fliped.to(device))
                embeddings[idx : idx + batch_size] = l2_norm(emb_batch)
            else:
                embeddings[idx : idx + batch_size] = model(batch.to(device)).cpu()
    return embeddings


class PretrainedInsightFaceClassifier2:
    def __init__(self, device, tta=True, db=False):
        self.prototype_cache_path = (
            f'insighface2_celeba_prototype_cache_tta{tta}_db{db}.pt'
        )

        conf = get_config(training=False)
        self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(device)
        self.model.load_state_dict(
            torch.load(
                os.path.join(
                    INSIGHTFACE_ROOT, conf.save_path / 'model_{}'.format('ir_se50.pth')
                )
            )
        )
        self.model.eval()
        self.embedding_size = conf.embedding_size
        self.device = device
        self.resize_layer = nn.UpsamplingBilinear2d((112, 112))
        self.tta = tta

        if os.path.exists(self.prototype_cache_path):
            self.prototypes = torch.load(self.prototype_cache_path).to(device)
        else:
            # import ipdb; ipdb.set_trace()
            # Get Celeb-A data
            train_x, train_y, _, _ = get_celeba_dataset(
                'target' if not db else 'db', False
            )
            bs = 500
            embeddings = []
            for start in range(0, len(train_x), bs):
                batch = train_x[start : start + bs]
                with torch.no_grad():
                    e = self.embed(batch)
                embeddings.append(e)
            train_embeddings = torch.cat(embeddings)
            prototypes = torch.zeros(1000, train_embeddings.size(1))
            for c in range(1000):
                prototypes[c] = train_embeddings[train_y == c].mean(0)
            self.prototypes = prototypes.to(device)
            torch.save(self.prototypes.cpu(), self.prototype_cache_path)

    def embed(self, x):
        model = self.model
        tta = self.tta

        batch = x
        batch = batch[:, [2, 1, 0]]
        batch = self.resize_layer(batch)
        if tta:
            fliped = hflip_batch(batch)
            emb_batch = model(batch) + model(fliped)
            embeddings = l2_norm(emb_batch)
        else:
            embeddings = model(batch)
        return embeddings

    def z_to_logits(self, z):
        dists = euclidean_dist(z, self.prototypes)
        return -dists

    def logits(self, x):
        z = self.embed(x)
        logits = self.z_to_logits(z)
        return logits

    def acc(self, x, y):
        logits = self.logits(x)
        preds = torch.max(logits, 1)[1]
        acc = (preds.cpu() == y.cpu()).float().mean()
        return acc.item()


class PretrainedInsightFaceClassifier:
    def __init__(self, device, pad=False):
        self.prototype_cache_path = f'insighface_celeba_prototype_cache_pad{pad}.pt'

        conf = get_config(training=False)
        self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(device)
        self.model.load_state_dict(
            torch.load(
                os.path.join(
                    INSIGHTFACE_ROOT, conf.save_path / 'model_{}'.format('ir_se50.pth')
                )
            )
        )
        self.model.eval()
        self.embedding_size = conf.embedding_size
        self.device = device
        self.pad = pad

        if os.path.exists(self.prototype_cache_path):
            self.prototypes = torch.load(self.prototype_cache_path).to(device)
        else:
            # import ipdb; ipdb.set_trace()
            # Get Celeb-A data
            train_x, train_y, _, _ = get_celeba_dataset('target', crop=pad)
            # Reverse RGB
            train_x = train_x[:, [2, 1, 0]]
            train_embeddings = insightface_fpass(
                train_x, device, self.model, self.embedding_size, pad=self.pad
            )
            prototypes = torch.zeros(1000, train_embeddings.size(1))
            for c in range(1000):
                prototypes[c] = train_embeddings[train_y == c].mean(0)
            self.prototypes = prototypes.to(device)
            torch.save(self.prototypes.cpu(), self.prototype_cache_path)

    def embed(self, x):
        return insightface_fpass(
            x.cpu(), self.device, self.model, self.embedding_size, pad=self.pad
        )

    def z_to_logits(self, z):
        dists = []
        for start in tqdm(range(0, len(z), 100), desc='comparing to prototypes'):
            dists.append(
                euclidean_dist(
                    z[start : start + 100].cuda().float(),
                    self.prototypes.cuda().float(),
                )
            )
        dists = torch.cat(dists)
        return -dists

    def logits(self, x):
        z = self.embed(x)
        logits = self.z_to_logits(z)
        return logits

    def acc(self, x, y):
        logits = self.logits(x)
        preds = torch.max(logits, 1)[1]
        acc = (preds.cpu() == y.cpu()).float().mean()
        return acc.item()


class FinetunednsightFaceClassifier(nn.Module):
    def __init__(
        self, device, L=3, eval_mode=False, pad=False, normalize_embedding=True
    ):
        assert L > 0
        super(FinetunednsightFaceClassifier, self).__init__()

        conf = get_config(training=False)
        self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(device)
        self.model.load_state_dict(
            torch.load(
                os.path.join(
                    INSIGHTFACE_ROOT, conf.save_path / 'model_{}'.format('ir_se50.pth')
                )
            )
        )
        self.embedding_size = conf.embedding_size
        self.device = device
        self.normalize_embedding = normalize_embedding
        self.model_unnorm = None
        # if not self.normalize_embedding:
        #     self.model_unnorm = deepcopy(self.model)
        #     self.model_unnorm.output_layer = self.model_unnorm.output_layer[:-1]

        # Decoder
        H = self.embedding_size
        layers = []
        for l in range(L):
            layers.append(nn.BatchNorm1d(H))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(H, 1000 if l == L - 1 else H))
        self.decoder = nn.Sequential(*layers)

        self.eval_mode = eval_mode
        self.pad = pad

    def embed(self, x):
        if not self.eval_mode:
            return self.model(x)
        else:
            with torch.no_grad():
                if self.pad:
                    x = padto112_batch(x)
                else:
                    x = resize112_batch(x)
                if not self.normalize_embedding:
                    if self.model_unnorm is None:
                        self.model_unnorm = deepcopy(self.model)
                        self.model_unnorm.output_layer = self.model_unnorm.output_layer[
                            :-1
                        ]
                        self.model_unnorm.normalize_output = False
                    model = self.model_unnorm
                else:
                    model = self.model
                x = torch.cat(
                    [model(x[start : start + 100]) for start in range(0, len(x), 100)]
                )
                return x

    def embed_img(self, x):
        return self.embed(x)

    def z_to_logits(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.embed(x)
        logits = self.z_to_logits(z)
        return logits

    def logits(self, x):
        return self.forward(x)

    def acc(self, x, y):
        logits = self.logits(x)
        preds = torch.max(logits, 1)[1]
        acc = (preds.cpu() == y.cpu()).float().mean()
        return acc.item()


def main_use_class():

    # Get Celeb-A data
    train_x, train_y, test_x, test_y = get_celeba_dataset('target', crop=False)
    # Reverse RGB
    test_x = test_x[:, [2, 1, 0]]  # if not, acc drops by 10%
    # Get model
    model = PretrainedInsightFaceClassifier('cuda:0', pad=False)
    logits = model.logits(test_x)
    preds = torch.max(logits, 1)[1]
    acc = (preds.cpu() == test_y.cpu()).float().mean()
    print(acc.item())


def main_use_class2():

    # Get Celeb-A data
    train_x, train_y, test_x, test_y = get_celeba_dataset('target', crop=False)
    # Reverse RGB
    test_x = test_x[:, [2, 1, 0]]  # if not, acc drops by 10%
    # Get model
    model = PretrainedInsightFaceClassifier2('cuda:0', tta=False, db=False)
    bs = 500
    logits = []
    for start in range(0, len(test_x), bs):
        with torch.no_grad():
            logits_ = model.logits(test_x[start : start + bs])
        logits.append(logits_)
    logits = torch.cat(logits)
    preds = torch.max(logits, 1)[1]
    acc = (preds.cpu() == test_y.cpu()).float().mean()
    print(acc.item())


def train(
    args, model, device, train_loader, optimizers, epoch, iteration_logger, trim=False
):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = DiffAugment(data / 2 + 0.5, args.augment).clamp(0, 1) * 2 - 1
        if trim:
            data = trim_batch(data)
        for _, optimizer in optimizers.items():
            optimizer.zero_grad()
        output = model(data)
        output = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        for _, optimizer in optimizers.items():
            optimizer.step()

        # Acc
        acc = (output.max(-1)[1] == target).float().mean().item()

        # Log
        if batch_idx % 50 == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

            iteration_logger.writerow(
                {
                    'global_iteration': batch_idx + len(train_loader) * epoch,
                    'train_acc': acc,
                    'train_loss': loss.item(),
                }
            )
            plot_csv(
                iteration_logger.filename,
                os.path.join(args.output_dir, 'iteration_plots.jpeg'),
            )

    # Sanity check: vis data
    if epoch == 1:
        vutils.save_image(
            data[:64], '%s/train_batch.jpeg' % (args.output_dir), normalize=True, nrow=8
        )


def test(args, model, device, test_loader, epoch=-1, trim=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if trim:
                data = trim_batch(data)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Sanity check: vis data
    if epoch == 1:
        vutils.save_image(
            data[:64], '%s/test_batch.jpeg' % (args.output_dir), normalize=True, nrow=8
        )

    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return 100.0 * correct / len(test_loader.dataset)


def main_finetune(args):
    device = 'cuda:0'
    batch_size = args.batch_size
    # Logging
    epoch_fieldnames = ['global_iteration', 'test_acc']
    epoch_logger = CSVLogger(
        every=1,
        fieldnames=epoch_fieldnames,
        filename=os.path.join(args.output_dir, 'epoch_log.csv'),
    )

    iteration_fieldnames = ['global_iteration', 'train_acc', 'train_loss']
    iteration_logger = CSVLogger(
        every=1,
        fieldnames=iteration_fieldnames,
        filename=os.path.join(args.output_dir, 'iteration_log.csv'),
    )

    # Get Celeb-A data
    train_x, train_y, test_x, test_y = get_celeba_dataset('target')
    # Reverse RGB
    train_x = train_x[:, [2, 1, 0]]
    test_x = test_x[:, [2, 1, 0]]
    # Preprocess & Augment data
    train_x = resize112_batch(train_x)
    train_x = torch.cat([hflip_batch(train_x), train_x])
    train_y = torch.cat([train_y, train_y])
    test_x = resize112_batch(test_x)
    test_x = torch.cat([hflip_batch(test_x), test_x])
    test_y = torch.cat([test_y, test_y])
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_x, test_y),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    # Get model
    model = FinetunednsightFaceClassifier('cuda:0', args.decoder_layers)
    model.to(device)
    optimizers = {
        'backbone': torch.optim.SGD(
            list(model.model.parameters()),
            lr=args.lr if not args.ttlr else args.lr * 0.1,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
        ),
        'decoder': torch.optim.SGD(
            list(model.decoder.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
        ),
    }

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizers, epoch, iteration_logger)
        test_acc = test(args, model, device, test_loader, epoch=epoch)

        epoch_logger.writerow(
            {
                'global_iteration': epoch,
                'test_acc': test_acc,
            }
        )
        plot_csv(
            epoch_logger.filename, os.path.join(args.output_dir, 'epoch_plots.jpeg')
        )

        torch.save(model.state_dict(), os.path.join(args.output_dir, "ckpt.pt"))
        if epoch in [1, 2, 5, 10, 20, 50]:
            torch.save(
                model.state_dict(), os.path.join(args.output_dir, f"ckpt_e{epoch}.pt")
            )


def main_finetune_crop(args):
    device = 'cuda:0'
    batch_size = args.batch_size
    # Logging
    epoch_fieldnames = ['global_iteration', 'test_acc']
    epoch_logger = CSVLogger(
        every=1,
        fieldnames=epoch_fieldnames,
        filename=os.path.join(args.output_dir, 'epoch_log.csv'),
    )

    iteration_fieldnames = ['global_iteration', 'train_acc', 'train_loss']
    iteration_logger = CSVLogger(
        every=1,
        fieldnames=iteration_fieldnames,
        filename=os.path.join(args.output_dir, 'iteration_log.csv'),
    )

    # Get Celeb-A data
    train_x, train_y, test_x, test_y = get_celeba_dataset('target', crop=False)
    # Reverse RGB
    train_x = train_x[:, [2, 1, 0]]
    test_x = test_x[:, [2, 1, 0]]
    # Preprocess data
    train_x = resize112_batch(train_x)
    test_x = resize112_batch(test_x)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_x, test_y),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    # Visualize Data
    x, y = iter(train_loader).next()
    vutils.save_image(
        x[:64], '%s/train_x.jpeg' % (args.output_dir), normalize=True, nrow=8
    )
    vutils.save_image(
        trim_batch(DiffAugment(x[:64] / 2 + 0.5, 'color')[:, [2, 1, 0]] * 2 - 1),
        '%s/train_x.jpeg' % (args.output_dir),
        normalize=True,
        nrow=8,
    )
    vutils.save_image(
        trim_batch(DiffAugment(x[:64] / 2 + 0.5, 'translation')[:, [2, 1, 0]] * 2 - 1),
        '%s/train_x-t.jpeg' % (args.output_dir),
        normalize=True,
        nrow=8,
    )
    vutils.save_image(
        trim_batch(
            DiffAugment(x[:64] / 2 + 0.5, 'color')[:, [2, 1, 0]].clamp(0, 1) * 2 - 1
        ),
        '%s/train_x-c.jpeg' % (args.output_dir),
        normalize=True,
        nrow=8,
    )
    vutils.save_image(
        trim_batch(DiffAugment(x[:64] / 2 + 0.5, 'cutout')[:, [2, 1, 0]] * 2 - 1),
        '%s/train_x-o.jpeg' % (args.output_dir),
        normalize=True,
        nrow=8,
    )
    vutils.save_image(
        trim_batch(DiffAugment(x[:64] / 2 + 0.5, 'cutout4')[:, [2, 1, 0]] * 2 - 1),
        '%s/train_x-o4.jpeg' % (args.output_dir),
        normalize=True,
        nrow=8,
    )
    vutils.save_image(
        trim_batch(
            DiffAugment(x[:64] / 2 + 0.5, 'cutout4,cutout4')[:, [2, 1, 0]] * 2 - 1
        ),
        '%s/train_x-o4o4.jpeg' % (args.output_dir),
        normalize=True,
        nrow=8,
    )
    vutils.save_image(
        trim_batch(
            DiffAugment(x[:64] / 2 + 0.5, 'translation,color,cutout4,cutout4')[
                :, [2, 1, 0]
            ]
            * 2
            - 1
        ),
        '%s/train_x-tco4o4.jpeg' % (args.output_dir),
        normalize=True,
        nrow=8,
    )

    # Get model
    model = FinetunednsightFaceClassifier('cuda:0', args.decoder_layers, pad=True)
    model.to(device)
    optimizers = {
        'backbone': torch.optim.SGD(
            list(model.model.parameters()),
            lr=args.lr if not args.ttlr else args.lr * 0.1,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
        ),
        'decoder': torch.optim.SGD(
            list(model.decoder.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
        ),
    }

    for epoch in range(1, args.epochs + 1):
        train(
            args,
            model,
            device,
            train_loader,
            optimizers,
            epoch,
            iteration_logger,
            trim=True,
        )
        test_acc = test(args, model, device, test_loader, epoch=epoch, trim=True)

        epoch_logger.writerow(
            {
                'global_iteration': epoch,
                'test_acc': test_acc,
            }
        )
        plot_csv(
            epoch_logger.filename, os.path.join(args.output_dir, 'epoch_plots.jpeg')
        )

        torch.save(model.state_dict(), os.path.join(args.output_dir, "ckpt.pt"))
        if epoch in [1, 2, 5, 10, 20, 50]:
            torch.save(
                model.state_dict(), os.path.join(args.output_dir, f"ckpt_e{epoch}.pt")
            )


if __name__ == '__main__':
    # device = 'cuda:0'
    # resize_layer = nn.UpsamplingBilinear2d((112,112))
    # model = PretrainedInsightFaceClassifier('cuda:0',pad=False)
    # x = torch.randn(100,3,112,112).to(device)
    # z = model.logits(x)
    # x = torch.randn(100,3,64,64).to(device)
    # z = model.logits(resize_layer(x))
    # import ipdb; ipdb.set_trace()
    main_use_class2()

    # main_reproduce_other_repo()
    # main()
    # main_use_class()
    import ipdb

    ipdb.set_trace()
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        metavar='N',
        help='input batch size for training (default: 64)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=14,
        metavar='N',
        help='number of epochs to train (default: 14)',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1.0,
        metavar='LR',
        help='learning rate (default: 1.0)',
    )
    parser.add_argument('--output_dir', required=True, help='')
    parser.add_argument('--eval', type=int, default=0)
    parser.add_argument('--augment', nargs='?', const='', type=str, default='')
    parser.add_argument('--decoder_layers', type=int, default=3)
    parser.add_argument('--ttlr', type=int, default=0)
    parser.add_argument('--ckpt_every', type=int, default=1)
    parser.add_argument('--crop', type=int, default=0)
    args = parser.parse_args()

    mkdir(args.output_dir)
    args.jobid = os.environ['SLURM_JOB_ID'] if 'SLURM_JOB_ID' in os.environ else -1
    utils.save_args(args, os.path.join(args.output_dir, f'args.json'))
    if args.crop:
        main_finetune_crop(args)
    else:
        main_finetune(args)
