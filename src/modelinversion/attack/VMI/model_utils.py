import torch
import matplotlib.pylab as plt
import numpy as np
import os
import yaml
from tqdm import tqdm
import itertools
import json
from types import SimpleNamespace

# local imports
import nets
 
from classify_mnist import Net, ResNetCls, ResNetCls1, VGG, get_model
from utils import get_item, euclidean_dist, gaussian_logp
from fid import calculate_frechet_distance


def _load_cls(dataset, cls_path, device):
    with open(os.path.join(os.path.split(cls_path)[0], 'args.json'), 'r') as f:
        model_args = json.load(f)
    # if 'model' in model_args:
    #     if model_args['model'] == 'ResNetCls1':
    #         if model_args['dataset'] == 'cifar10':
    #             # C,H,W = 3,32,32
    #             classifier = ResNetCls1(3, zdim=model_args['latent_dim']).to(device)
    #         elif model_args['dataset'].startswith('celeba'):
    #             # C,H,W = 3,64,64
    #             classifier = ResNetCls1(3, zdim=model_args['latent_dim'], imagesize=64,nclass=1000,resnetl=model_args['resnetl'], dropout=model_args['dropout']).to(device)
    #         else:
    #             # C,H,W = 1,32,32
    #             classifier = ResNetCls1(1, zdim=model_args['latent_dim']).to(device)
    #     elif model_args['model'] == 'vgg':
    #         if model_args['dataset'].startswith('celeba'):
    #             # C,H,W = 3,64,64
    #             classifier =  VGG(zdim=model_args['latent_dim'], nclass=1000, dropout=model_args['dropout']).to(device)
    classifier = get_model(SimpleNamespace(**model_args), device)[0]

    # else: # OLD MODELS...
    #     if dataset == 'mnist':
    #       # C,H,W = 1,model_args.imageSize,model_args.imageSize
    #       classifier = Net(nc=1, nz=128).to(device)
    #     elif dataset in ['cifar10','cifar0to4', 'svhn']:
    #       # C,H,W = 3,model_args.imageSize,model_args.imageSize
    #       classifier = ResNetCls().to(device)
    #     else:
    #       raise ValueError()

    classifier.load_state_dict(torch.load(cls_path))
    classifier.eval()
    return classifier, model_args


def load_cls_z_to_lsm(dataset, cls_path, device):
    classifier, model_args = _load_cls(dataset, cls_path, device)
    return lambda z: classifier.z_to_lsm(z)


def load_cls_embed(dataset, cls_path, device, classify=False, logits=False):
    classifier, model_args = _load_cls(dataset, cls_path, device)
    print(classifier)
    # Which output head
    assert not (classify and logits)
    if classify:
        func = classifier
    elif logits:
        func = classifier.logits
    else:
        func = classifier.embed_img

    H = W = 64  # model_args['imageSize']
    if dataset in ['mnist']:
        C = 1
    elif dataset in ['cifar10', 'cifar0to4', 'svhn']:
        C = 3
    elif dataset.startswith('celeba') or dataset in ['pubfig83', 'cfw']:
        C = 3

    if dataset == 'chestxray':
        C = 1
        H = W = 128

    if dataset == 'mnist':
        def extract_feat(x):
            assert x.min() >= 0  # in case passing in x in [-1,1] by accident
            return func((x.view(x.size(0), C, H, W) - 0.1307) / 0.3081)
    elif dataset == 'cifar10':
        def extract_feat(x):
            assert x.min() >= 0  # in case passing in x in [-1,1] by accident
            x = x.view(x.size(0), C, H, W).clone()
            x[:, 0].add_(-0.4914).mul_(1 / 0.2023)
            x[:, 1].add_(-0.4822).mul_(1 / 0.1994)
            x[:, 2].add_(-0.4465).mul_(1 / 0.2010)
            return func(x)
    elif dataset == 'cifar0to4':
        def extract_feat(x):
            assert x.min() >= 0  # in case passing in x in [-1,1] by accident
            x = x.view(x.size(0), C, H, W).clone()
            x[:, 0].add_(-0.4907).mul_(1 / 0.2454)
            x[:, 1].add_(-0.4856).mul_(1 / 0.2415)
            x[:, 2].add_(-0.4509).mul_(1 / 0.2620)
            return func(x)
    elif dataset == 'svhn':
        def extract_feat(x):
            assert x.min() >= 0  # in case passing in x in [-1,1] by accident
            return func((x.view(x.size(0), C, H, W) - .5) / 0.5)
    elif dataset.startswith('celeba') or dataset in ['pubfig83', 'cfw']:
        def extract_feat(x, mb=100):
            assert x.min() >= 0  # in case passing in x in [-1,1] by accident
            zs = []
            for start in range(0, len(x), mb):
                _x = x[start:start + mb]
                zs.append(func((_x.view(_x.size(0), C, H, W) - .5) / 0.5))
            return torch.cat(zs)
    elif dataset == 'chestxray':
        def extract_feat(x, mb=100):
            assert x.min() >= 0  # in case passing in x in [-1,1] by accident
            zs = []
            for start in range(0, len(x), mb):
                _x = x[start:start + mb]
                zs.append(func((_x.view(_x.size(0), C, H, W) - .5) / 0.5))
            return torch.cat(zs)

    return extract_feat


def parse_kwargs(s):
    r = {}
    if s == '':
        return r
    for item in s.split(','):
        k, tmp = item.split(":")
        t, v = tmp.split(".")
        if t == 'i':
            v = int(v)
        elif t == 'f':
            v = float(v)
        elif t == 's':
            pass
        else:
            raise
        r[k] = v
    return r


def instantiate_generator(args, device):
    if args.gen in ['basic', 'conditional', 'conditional_no_embed']:
        generator = nets.ConditionalGenerator(args.imageSize, args.nz, args.ngf, args.nc, args.n_conditions, args.gen in ['conditional', 'conditional_no_embed'], args.g_sn, args.g_z_scale, args.g_conditioning_method, args.gen != 'conditional_no_embed',
                                              norm=args.g_norm, cdim=args.cdim).to(device)
    elif args.gen in ['secret', 'secret-conditional']:
        generator = nets.ConditionalGeneratorSecret(args.imageSize, args.nz, args.ngf, args.nc, args.n_conditions, args.gen == 'secret-conditional', args.g_sn, args.g_z_scale, args.g_conditioning_method, args.gen != 'conditional_no_embed',
                                                    norm=args.g_norm, cdim=args.cdim).to(device)
    elif args.gen == 'toy':
        generator = nets.ConditionalGeneratorToy(args.imageSize, args.nz, args.ngf, args.nc, args.n_conditions,
                                                 args.use_labels, args.g_sn, args.g_z_scale, args.g_conditioning_method).to(device)
    else:
        raise ValueError()
    return generator


def instantiate_discriminator(args, index2class, device):
    disc_config = yaml.load(open(f'disc_config/{args.disc_config}', 'r'))
    disc_config['kwargs']['imgSize'] = args.imageSize
    disc_config['kwargs']['nc'] = args.nc
    disc_config['kwargs']['cdim'] = args.cdim
    # Override kwargs
    override_kwargs = parse_kwargs(args.disc_kwargs)
    for k in override_kwargs:
        disc_config['kwargs'][k] = override_kwargs[k]
    if args.use_labels:
        assert disc_config['kwargs']['is_conditional']
        assert args.n_conditions == disc_config['kwargs']['n_conditions']
        disc_config['kwargs']['index2class'] = index2class
    # import ipdb; ipdb.set_trace()
    discriminator = eval(
        'nets.'+disc_config['name'])(**disc_config['kwargs']).to(device)
    print(discriminator)
    print(dir(discriminator))
    return discriminator




def compute_gmm_loglikelihood(samples, means, stds):
    logp_xc = []
    for m, s in tqdm(zip(means, stds), desc='gmm'):
        logp_xc.append(gaussian_logp(m, s.log(), samples).sum(-1))
    logp_xc = torch.stack(logp_xc)
    logp_x = torch.logsumexp(logp_xc, dim=0) - np.log(logp_xc.size(0))
    return logp_x


def estimate_params(x, y):
    means = []
    stds = []
    for c in range(10):
        idx = c == y
        chunk = x[idx]
        if len(chunk) <= 1:  # undefined std
            continue
        means.append(chunk.mean(0, keepdim=True).detach())
        stds.append(chunk.std(0, keepdim=True).detach())
    return means, stds


def compute_divergences(x1, y1, x2, y2, compute_all=False):
    means1, stds1 = estimate_params(x1, y1)
    means2, stds2 = estimate_params(x2, y2)

    def compute_kl(params_p, params_q, samples_p):
        logp = compute_gmm_loglikelihood(samples_p, params_p[0], params_p[1])
        logq = compute_gmm_loglikelihood(samples_p, params_q[0], params_q[1])
        return (logp - logq).mean()

    def compute_jsd(params_p, params_q, samples_p, samples_q):
        # KL(P || .5(P+Q))
        logp = compute_gmm_loglikelihood(samples_p, params_p[0], params_p[1])
        logq = compute_gmm_loglikelihood(samples_p, params_q[0], params_q[1])
        kl_pm = (
            logp - (torch.logsumexp(torch.stack([logp, logq]), 0) - np.log(2))).mean()
        # KL(Q || .5(P+Q))
        logp = compute_gmm_loglikelihood(samples_q, params_p[0], params_p[1])
        logq = compute_gmm_loglikelihood(samples_q, params_q[0], params_q[1])
        kl_qm = (
            logq - (torch.logsumexp(torch.stack([logp, logq]), 0) - np.log(2))).mean()
        return 0.5*(kl_pm + kl_qm)

    kl_12 = kl_21 = jsd = frechet = -1

    # Frechet Distance
    npx1 = x1.detach().cpu().numpy()
    npx2 = x2.detach().cpu().numpy()
    mu1 = np.mean(npx1, axis=0)
    sigma1 = np.cov(npx1, rowvar=False)
    mu2 = np.mean(npx2, axis=0)
    sigma2 = np.cov(npx2, rowvar=False)
    frechet = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    if compute_all:
        kl_12 = compute_kl((means1, stds1), (means2, stds2), x1)
        kl_21 = compute_kl((means2, stds2), (means1, stds1), x2)
        jsd = compute_jsd((means1, stds1), (means2, stds2), x1, x2)
    return {
        'kl_12': kl_12,
        'kl_21': kl_21,
        'jsd': jsd,
        'frechet': frechet
    }


def compute_gmm_loglikelihood(samples, means, stds):
    logp_xc = []
    for m, s in tqdm(zip(means, stds), desc='gmm'):
        logp_xc.append(gaussian_logp(m, s.log(), samples).sum(-1))
    logp_xc = torch.stack(logp_xc)
    logp_x = torch.logsumexp(logp_xc, dim=0) - np.log(logp_xc.size(0))
    return logp_x


def estimate_params(x, y):
    means = []
    stds = []
    for c in range(10):
        idx = c == y
        chunk = x[idx]
        if len(chunk) <= 1:  # undefined std
            continue
        means.append(chunk.mean(0, keepdim=True).detach())
        stds.append(chunk.std(0, keepdim=True).detach())
    return means, stds


def compute_divergences(x1, y1, x2, y2, compute_all=False):
    means1, stds1 = estimate_params(x1, y1)
    means2, stds2 = estimate_params(x2, y2)

    def compute_kl(params_p, params_q, samples_p):
        logp = compute_gmm_loglikelihood(samples_p, params_p[0], params_p[1])
        logq = compute_gmm_loglikelihood(samples_p, params_q[0], params_q[1])
        return (logp - logq).mean()

    def compute_jsd(params_p, params_q, samples_p, samples_q):
        # KL(P || .5(P+Q))
        logp = compute_gmm_loglikelihood(samples_p, params_p[0], params_p[1])
        logq = compute_gmm_loglikelihood(samples_p, params_q[0], params_q[1])
        kl_pm = (
            logp - (torch.logsumexp(torch.stack([logp, logq]), 0) - np.log(2))).mean()
        # KL(Q || .5(P+Q))
        logp = compute_gmm_loglikelihood(samples_q, params_p[0], params_p[1])
        logq = compute_gmm_loglikelihood(samples_q, params_q[0], params_q[1])
        kl_qm = (
            logq - (torch.logsumexp(torch.stack([logp, logq]), 0) - np.log(2))).mean()
        return 0.5*(kl_pm + kl_qm)

    kl_12 = kl_21 = jsd = frechet = -1

    # Frechet Distance
    npx1 = x1.detach().cpu().numpy()
    npx2 = x2.detach().cpu().numpy()
    mu1 = np.mean(npx1, axis=0)
    sigma1 = np.cov(npx1, rowvar=False)
    mu2 = np.mean(npx2, axis=0)
    sigma2 = np.cov(npx2, rowvar=False)
    frechet = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    if compute_all:
        kl_12 = compute_kl((means1, stds1), (means2, stds2), x1)
        kl_21 = compute_kl((means2, stds2), (means1, stds1), x2)
        jsd = compute_jsd((means1, stds1), (means2, stds2), x1, x2)
    return {
        'kl_12': kl_12,
        'kl_21': kl_21,
        'jsd': jsd,
        'frechet': frechet
    }


class kNNManifold:
    def __init__(self, x):
        assert len(x.shape) == 2
        assert x.shape[0] > 1
        BS = 500
        self.x = x
        # compute all pair-wise distances
        with torch.no_grad():
            pairwise_dists = euclidean_dist(x, x)
        self.sorted_pairwise_dists = torch.sort(pairwise_dists, 1)[0]

    def compute_y_in_kmanifold(self, y, k, factor=1):
        assert factor == 1
        # Compute radiuses based on k
        r = self.sorted_pairwise_dists[:, k]

        # Compute pairwise distances between y, x
        pairwise_dists = euclidean_dist(y, self.x)

        # Is the closest neighbor within the radius?
        min_dists, min_x_idx = pairwise_dists.min(1)
        covered = (min_dists <= factor * r[min_x_idx]).float()
        return torch.mean(covered)

    def compute_fraction_manifold_covered(self, y, k, factor=1):
        assert factor == 1
        Ny = y.size(0)
        # Compute radiuses based on k
        r = self.sorted_pairwise_dists[:, k]

        # Compute pairwise distances between y, x
        pairwise_dists = euclidean_dist(y, self.x)  # (Ny, Nx)

        return ((pairwise_dists < r[None].repeat(Ny, 1)).float().sum(0) > 0).float().mean()

    def compute_density(self, y, k, factor=1):
        assert factor == 1
        Ny = y.size(0)
        # Compute radiuses based on k
        r = self.sorted_pairwise_dists[:, k]

        # Compute pairwise distances between y, x
        pairwise_dists = euclidean_dist(y, self.x)  # (Ny, Nx)

        repeated_r = r[None].repeat(Ny, 1)
        count_sum = (pairwise_dists < repeated_r).float().sum()

        return (1 / (Ny * k)) * count_sum


class GaussianManifold:
    def __init__(self, x):
        assert len(x.shape) == 2
        assert x.shape[0] > 1
        self.x = x
        # compute all pair-wise distances
        self.mean = x.mean(0)
        self.r = x.std(1).mean()

    def compute_y_in_kmanifold(self, y, k, factor=1):
        # Compute pairwise distances between y, x
        dists = torch.pow(y - self.mean, 2).sum(-1)

        # Is the closest neighbor within the radius?
        covered = (dists <= factor * self.r).float()
        return torch.mean(covered)


class PRCD:
    def __init__(self, run_feature_extract, target_data_x, manifold_type='knn'):
        self.run_feature_extract = run_feature_extract
        self.target_data_x = target_data_x
        self.manifold_type = manifold_type

        # Cache features
        with torch.no_grad():
            bs = 500
            self.target_data_z = torch.cat([
                self.run_feature_extract(
                    self.target_data_x[start:start + bs]
                ).cpu() for start in range(0, len(self.target_data_x), bs)])
            if self.manifold_type == 'knn':
                self.real_manifold = kNNManifold(self.target_data_z.cuda())
            elif self.manifold_type == 'gaussian':
                self.real_manifold = GaussianManifold(
                    self.target_data_z.cuda())

    def evaluate(self, sample_x):
        with torch.no_grad():
            bs = 500
            x1 = torch.cat([
                self.run_feature_extract(
                    sample_x[start:start + bs]
                ) for start in range(0, len(sample_x), bs)])

        # Precision / Recall
        D = {}
        for k in [5, 10]:
            if self.manifold_type == 'knn':
                fake_manifold = kNNManifold(x1)
            elif self.manifold_type == 'gaussian':
                fake_manifold = GaussianManifold(x1)
            real_manifold = self.real_manifold
            precision = real_manifold.compute_y_in_kmanifold(x1, k=k, factor=1)
            recall = fake_manifold.compute_y_in_kmanifold(
                self.target_data_z.cuda(), k=k, factor=1)
            D[f'precision@{k}'] = get_item(precision)
            D[f'recall@{k}'] = get_item(recall)

        # Coverage / Density
        for k in [5, 10]:
            fake_manifold = kNNManifold(x1)
            real_manifold = self.real_manifold
            coverage = real_manifold.compute_fraction_manifold_covered(
                x1, k=k, factor=1)
            density = real_manifold.compute_density(
                x1, k=k, factor=1)
            D[f'coverage@{k}'] = get_item(coverage)
            D[f'density@{k}'] = get_item(density)

        return {
            'precision@5': D['precision@5'],
            'recall@5': D['recall@5'],
            'precision@10': D['precision@10'],
            'recall@10': D['recall@10'],
            'coverage@5': D['coverage@5'],
            'density@5': D['density@5'],
            'coverage@10': D['coverage@10'],
            'density@10': D['density@10'],
        }


class TargetDatasetReconstructionEvaluation:
    def __init__(self, evaluation_classifier, target_data_x, target_data_y, bgr=True, run_target_feat_eval=True, k=3, factor=1, manifold_type='knn'):
        self.evaluation_classifier = evaluation_classifier
        self.target_data_x = target_data_x
        self.target_data_y = target_data_y
        self.bgr = bgr
        self.run_target_feat_eval = run_target_feat_eval
        self.k = k
        self.factor = factor
        self.manifold_type = manifold_type

        # Cache features
        if run_target_feat_eval:
            with torch.no_grad():
                bs = 500
                self.target_data_z = torch.cat([
                    self.evaluation_classifier.embed(
                        self.target_data_x[start:start+bs, [2, 1, 0]
                                           ] if self.bgr else self.target_data_x[start:start+bs]
                    ).cpu() for start in range(0, len(self.target_data_x), bs)])
                if self.manifold_type == 'knn':
                    self.real_manifold = kNNManifold(self.target_data_z.cuda())
                elif self.manifold_type == 'gaussian':
                    self.real_manifold = GaussianManifold(
                        self.target_data_z.cuda())

    def compute_eval_embeds(self, x):
        return self.evaluation_classifier.embed(x[:, [2, 1, 0]] if self.bgr else((x / 2 + .5) - 0.1307) / 0.3081)

    def compute_eval_logits(self, x):
        return self.evaluation_classifier.logits(x[:, [2, 1, 0]] if self.bgr else((x / 2 + .5) - 0.1307) / 0.3081)

    def compute_eval_preds(self, x):
        return self.compute_eval_logits(x).max(1)[1]

    def get_eval_preds(self, x):
        return self.compute_eval_preds(x)

    def evaluate(self, sample_x, sample_y, target_classifier, run_frechet=True):
        assert sample_y is None or target_classifier is None
        N = len(sample_x)
        if sample_y is None:
            sample_y = target_classifier(sample_x / 2 + .5).max(1)[1]
        sample_y = sample_y.cuda()

        x1 = self.evaluation_classifier.embed(
            sample_x[:, [2, 1, 0]] if self.bgr else ((sample_x / 2 + .5) - 0.1307) / 0.3081)
        # Frechet Distance
        if self.run_target_feat_eval:
            x2 = self.target_data_z
            npx1 = x1.detach().cpu().numpy()
            npx2 = x2.detach().cpu().numpy()
            if len(sample_x) == 1 or not run_frechet:
                frechet = -1
            else:
                mu1 = np.mean(npx1, axis=0)
                sigma1 = np.cov(npx1, rowvar=False)
                mu2 = np.mean(npx2, axis=0)
                sigma2 = np.cov(npx2, rowvar=False)
                frechet = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

        # Acc
        logits = self.evaluation_classifier.z_to_logits(x1)
        preds = torch.max(logits, 1)[1]
        acc = (preds.cpu() == sample_y.cpu()).float().mean().item()
        top5 = torch.topk(logits, k=5, dim=1)[1]
        top5_acc = np.mean([y in t for y, t in zip(
            sample_y.cpu().numpy(), top5.cpu().numpy())])

        # (Avg) Feature Distance
        if self.run_target_feat_eval:
            m = x2.mean(0, keepdim=True).cuda()
            l2 = torch.sqrt(torch.pow(x1 - m, 2)).mean()
            cos = torch.cosine_similarity(x1, m).mean()

            # Precision / Recall
            D = {}
            for k in [5, 10]:
                if self.manifold_type == 'knn':
                    fake_manifold = kNNManifold(x1)
                elif self.manifold_type == 'gaussian':
                    fake_manifold = GaussianManifold(x1)
                real_manifold = self.real_manifold
                precision = real_manifold.compute_y_in_kmanifold(
                    x1, k=k, factor=self.factor)
                recall = fake_manifold.compute_y_in_kmanifold(
                    self.target_data_z.cuda(), k=k, factor=self.factor)
                D[f'precision@{k}'] = get_item(precision)
                D[f'recall@{k}'] = get_item(recall)

            # Coverage / Density
            for k in [5, 10]:
                fake_manifold = kNNManifold(x1)
                real_manifold = self.real_manifold
                coverage = real_manifold.compute_fraction_manifold_covered(
                    x1, k=k, factor=self.factor)
                density = fake_manifold.compute_density(
                    self.target_data_z.cuda(), k=k, factor=self.factor)
                D[f'coverage@{k}'] = get_item(precision)
                D[f'density@{k}'] = get_item(recall)

        if self.run_target_feat_eval:
            return {
                'acc': acc,
                'top5_acc': top5_acc,
                'frechet': frechet,
                'feature-l2-dist': get_item(l2),
                'feature-cos-sim': get_item(cos),
                'precision@5': D['precision@5'],
                'recall@5': D['recall@5'],
                'precision@10': D['precision@10'],
                'recall@10': D['recall@10'],
                'coverage@5': D['coverage@5'],
                'density@5': D['density@5'],
                'coverage@10': D['coverage@10'],
                'density@10': D['density@10'],
            }
        else:
            return {
                'acc': acc,
                'top5_acc': top5_acc,
            }

