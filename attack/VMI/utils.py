import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import init
from types import SimpleNamespace
import json
from tqdm import tqdm
import PIL.Image
from copy import deepcopy


def save_image_grid(img, fname, drange, grid_size, ids_to_prepend=None):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    if ids_to_prepend is not None:
        # Load number images
        nrow = gh
        assert len(ids_to_prepend) == nrow
        num_ims = np.concatenate([plt.imread(f'results/number_imgs/{num}.png') for num in ids_to_prepend], 0)
        num_ims = np.rint(num_ims * 255).clip(0, 255).astype(np.uint8)
        img = np.concatenate([num_ims, img], 1)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)


def torch_mvn_logp(x, m, C):
    """
    input
        x - (N, k) data matrix torch.Tensor
        m - (1, k) mean torch.Tensor
        C - (k, k) covariance torch.Tensor
    output
        (N, ) logp = N(x; m, C), torch.Tensor
    """
    assert len(x.shape) == 2
    assert x.shape[1] == m.shape[-1]
    assert m.shape[0] == 1
    assert m.shape[1] == C.shape[0] == C.shape[1]

    k = x.shape[1]
    Z = -(k/2) * np.log(2*np.pi) - (1/2) * torch.logdet(C)
    Cinv = torch.inverse(C)
    s = -(1/2) * (((x - m) @ Cinv) * (x - m)).sum(-1)
    return Z + s


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    return torch.pow(x.unsqueeze(1).expand(n, m, d) - y.unsqueeze(0).expand(n, m, d), 2).sum(2)


def save_checkpoint(args, state):
    torch.save(state, args.ckpt)


def maybe_load_checkpoint(args):
    if args.resume and os.path.exists(args.ckpt):
        return torch.load(args.ckpt)


def get_item(v):
    if type(v) == int or np.isscalar(v):
        return v
    elif isinstance(v, torch.Tensor):
        return v.item()
    else:
        raise


def gaussian_logp(mean, logstd, x, detach=False):
    """
    lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
            k = 1 (Independent)
            Var = logstd ** 2
    """
    c = np.log(2 * np.pi)
    v = -0.5 * (logstd * 2. + ((x - mean) ** 2) / torch.exp(logstd * 2.) + c)
    if detach:
        v = v.detach()
    return v


class Stats:
    def __init__(self):
        self.items = []

    def add(self, x):
        self.items.append(x)

    def avg(self):
        return np.mean(self.items)


def save_args(args, fp):
    if type(args) != dict:
        args = vars(args)
        args = deepcopy(args)

    # Remove Known Wrong Types (that don't need to be saved)
    if 'device' in args:
        args.pop('device')
    if 'print' in args:
        args.pop('print')

    # Check all values are basic types
    assert all(map(lambda v: type(v) in [int, float, str, bool], args.values()))
    return json.dump(args, open(fp, 'w'),
                     sort_keys=True, indent=4)


def load_args(fp):
    return SimpleNamespace(**json.load(open(fp, 'r')))


def generate_n_samples(generator, generator_args, device, N):
    all_samples = []
    for start in tqdm(range(0, N, generator_args.num_gen_images), desc='generate_n_samples'):
        with torch.no_grad():
            fake = generator(torch.randn(
                generator_args.num_gen_images, generator_args.nz, 1, 1, device=device))
        all_samples.append(fake)
    return torch.cat(all_samples, 0)


def plot_kde(samples, epoch, name, title='', cmap='Blues', save_path=None):
    samples = samples.cpu().numpy()
    sns.set(font_scale=2)
    f, ax = plt.subplots(figsize=(4, 4))
    sns.kdeplot(samples[:, 0], samples[:, 1], cmap=cmap,
                ax=ax, n_levels=20, shade=True)
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.axis('off')
    plt.title(title)
    if save_path is not None:
        plt.savefig(os.path.join(save_path, '{}_{}.pdf'.format(name, epoch)))
    plt.show()


def weights_init(model, method='N02', std=0.02):
    for m in model.modules():
        classname = m.__class__.__name__
        # if (classname.find('Conv') != -1 or
        #     classname.find('Linear') != -1 or
        #     classname.find('Embedding') != -1):
        if (isinstance(m, torch.nn.Conv2d)
            or isinstance(m, torch.nn.Linear)
                or isinstance(m, torch.nn.Embedding)):
            print(f'Initializing: {classname}')
            if method == 'ortho':
                init.orthogonal_(m.weight)
            elif method == 'N02':
                init.normal_(m.weight, 0, 0.02)
            elif method == 'N':
                init.normal_(m.weight, 0, std)
            elif method in ['glorot', 'xavier']:
                init.xavier_uniform_(m.weight)
            else:
                print('Init style not recognized...')
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


def tonp(x):
    if not isinstance(x, np.ndarray):
        x = x.detach().cpu().numpy()
    return x


def mkdir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def prepare_random_indices(x, eval_size):
    all_indices = np.arange(len(x))
    np.random.shuffle(all_indices)
    return torch.from_numpy(all_indices[:eval_size]).to(x.device)


def expand_stackedmnist(x):
    N, three, W, H = x.shape
    y = []
    for _x in x:
        _y = torch.zeros((3, W*2, H*2))
        _y[:, :W, :H] = _x
        _y[:, W:, :H] = _x[0].unsqueeze(0).repeat(3, 1, 1)
        _y[:, :W, H:] = _x[1].unsqueeze(0).repeat(3, 1, 1)
        _y[:, W:, W:] = _x[2].unsqueeze(0).repeat(3, 1, 1)
        y.append(_y)
    y = torch.stack(y)
    return y
