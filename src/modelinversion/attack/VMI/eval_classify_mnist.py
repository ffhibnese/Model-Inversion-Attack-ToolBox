from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from classify_mnist import Net
import numpy as np
import ipdb
import itertools
import matplotlib.pylab as plt
from scipy.stats import entropy
from tqdm import tqdm
from utils import tonp


def test(args, model, device, test_loader, tr=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if tr is not None:
                data = torch.stack([tr(x) for x in data])
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction='sum'
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def unstack_mnist(x, y):
    x1 = x.unsqueeze(1).permute(0, 1, 3, 2)
    y1 = torch.from_numpy(np.array([y % 10, y // 10 % 10, y // 100 % 10]))
    return x1, y1


def plot_digits(x, y, fpath):
    x = x[:25].detach().cpu().numpy()
    y = y[:25].detach().cpu().numpy()
    fig, axs = plt.subplots(5, 5, figsize=(10, 7))
    for n, ax in enumerate(itertools.chain(*axs)):
        ax.imshow(x[n][0])
        ax.set_title(str(y[n]))
    plt.tight_layout()
    plt.savefig(fpath, bbox_inches='tight', pad_inches=0)


def compute_is_from_preds(preds, splits):
    # Now compute the mean kl-div
    N = len(preds)
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits) : (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
    return np.mean(split_scores), np.std(split_scores)


def expand_pyx(ipyx):
    """
    :inputs:
      ipyx - (B,3,10)
    :outputs:
      pyx -  (B,1000)
    """
    pyx = np.zeros((len(ipyx), 1000))
    # loop below is fast enough: 73328.02it/s
    for n, _ipyx in tqdm(enumerate(ipyx)):
        a, b, c = _ipyx
        tmp = np.outer(a, b).ravel()
        pyx[n] = np.outer(tmp, c).ravel()

    assert np.abs(pyx.sum(-1).mean() - 1) < 1e-5
    return pyx


def compute_stackedmnist_stats(ipyx):
    """
    :inputs:
      ipyx - p(y|x), but represented as shape (B, 3, 10)
            p(y|x) is actually pyx.prod(-1) where  y \in [0,999]

    :outputs:
      n_modes - number of modes covered, int \in [1,1e3]
      modes - e.g. [0,1,100,300]
      brier - the Brier score, \in roughly [0,1]
      IS    - Inception Score, in this case in  \in [1,1e3]
    """
    ipyx = tonp(ipyx)
    pyx = expand_pyx(ipyx)
    predicted_modes = np.argmax(pyx, -1)
    modes = list(set(predicted_modes))
    n_modes = len(modes)
    # mean Brier
    predicted_onehot = np.eye(1000)[predicted_modes]
    brier = (1.0 / len(pyx)) * np.sqrt(  # (1./pyx.shape[1]) *
        np.power(pyx - predicted_onehot, 2).sum(-1)
    ).sum()
    IS = compute_is_from_preds(pyx, 1)[0]

    return n_modes, modes, brier, IS


def test_stacked_mnist(model, device, test_loader, tr, no_label=True):
    model.eval()
    test_loss = 0
    correct = 0
    pyx = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = zip(*[unstack_mnist(tr(x), y) for x, y in zip(data, target)])
            data, target = torch.stack(data).to(device), torch.stack(target).to(device)
            nB, nC, one, W, H = data.shape
            data = data.reshape(nB * nC, one, W, H)
            target = target.reshape(nB * nC)
            output = model(data)
            pyx.append(torch.softmax(output, -1).reshape(nB, nC, 10))
            if not no_label:
                test_loss += F.nll_loss(
                    output, target, reduction='sum'
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

    # plot_digits(data,target,'tmp.png')
    if not no_label:
        test_loss /= len(test_loader.dataset)
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )
    #
    pyx = torch.cat(pyx, 0)
    n_modes, modes, brier, IS = compute_stackedmnist_stats(pyx)
    print(n_modes, brier, IS)
    return n_modes, brier, IS


def make_loader(x, y=None):
    if y is None:
        y = torch.zeros(len(x)).long()
    dset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(
        dset, batch_size=1000, shuffle=False, num_workers=1, drop_last=False
    )
    return loader


class EvalStackedMNIST(object):
    """docstring for EvalStackedMNIST"""

    def __init__(self, device):
        super(EvalStackedMNIST, self).__init__()
        self.device = device
        self.pretrained_classifier = Net().to(device)
        sd = torch.load("mnist_cnn.pt")
        self.pretrained_classifier.load_state_dict(sd)
        self.tr = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(28),
                transforms.ToTensor(),
                transforms.Normalize((-0.5,), (2,)),  # undo previous normalize
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def eval_samples(self, samples):
        loader = make_loader(samples)
        # with torch.no_grad():
        return test_stacked_mnist(
            self.pretrained_classifier, self.device, loader, self.tr
        )


def main():

    # from train import check_dataset
    import data
    import ipdb

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        metavar='N',
        help='input batch size for training (default: 64)',
    )
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=1000,
        metavar='N',
        help='input batch size for testing (default: 1000)',
    )
    parser.add_argument(
        '--seed', type=int, default=1, metavar='S', help='random seed (default: 1)'
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='how many batches to wait before logging training status',
    )

    parser.add_argument(
        '--save-model',
        action='store_true',
        default=False,
        help='For Saving the current Model',
    )
    args = parser.parse_args()
    use_cuda = True

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '/scratch/gobi1/wangkuan/data/data/MNIST/MNIST',
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '/scratch/gobi1/wangkuan/data/data/MNIST/MNIST',
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs
    )

    model = Net().to(device)
    sd = torch.load("mnist_cnn.pt")
    model.load_state_dict(sd)

    test(args, model, device, train_loader)
    # test(args, model, device, test_loader)

    # # MNIST
    # dat = data.load_data('mnist','/scratch/gobi1/wangkuan/data/data/MNIST/MNIST' ,
    #                     device=device, imgsize=28, Ntrain=60000, Ntest=10000)

    # test_dataset = torch.utils.data.TensorDataset(dat['X_test'],dat['Y_test'])
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000,
    #                               shuffle=False, num_workers=1,
    #                               drop_last=False)
    # tr = transforms.Compose([
    #                        transforms.ToPILImage(),
    #                        transforms.Resize(28),
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((-.5,), (2,)), # undo previous normalize
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])
    # test(args, model, device, test_loader, tr)

    # StackedMNIST
    dat = data.load_data(
        'stackedmnist', 'data', device=device, imgsize=64, Ntrain=60000, Ntest=10000
    )
    eval_runner = EvalStackedMNIST(device)
    print(eval_runner.eval_samples(dat['X_test']))

    # test_loader = make_loader(dat['X_test'],dat['Y_test'])
    # train_loader = make_loader(dat['X_train'],dat['Y_train'])
    # r_loader = make_loader(dat['X_test'])
    # test_stacked_mnist( model, device, test_loader, tr)
    # test_stacked_mnist( model, device, r_loader, tr)
    # # print("Training set")
    # # test_stacked_mnist( model, device, train_loader, tr)


if __name__ == '__main__':
    main()
