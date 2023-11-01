import os
import numpy as np
import torch
from tqdm import tqdm
from eval_pretrained_face_classifier import PretrainedInsightFaceClassifier
from main_aux import PRCD
import torchvision.utils as vutils
import argparse
from collections import defaultdict
import pandas
from fid import run_fid, run_feature_extractor, postprocess
import matplotlib.pylab as plt
import data

device = 'cuda:0'


def _load_real_cache():
    cache_path = 'May18-celeba-target-100ids-cache.pt'
    if os.path.exists(cache_path):
        return torch.load(cache_path)
    else:
        dat = data.load_data('celeba-target')
        X = torch.cat([dat['X_train'], dat['X_test']])
        Y = torch.cat([dat['Y_train'], dat['Y_test']])
        xs = []
        ys = []
        for c in range(100):
            m = Y == c
            x = X[m]
            y = Y[m]
            xs.append(x)
            ys.append(y)
        xs = torch.cat(xs)
        ys = torch.cat(ys)
        torch.save((xs, ys), cache_path)
        return xs, ys


def _load_samples_pt(args, fprefix):
    fake, fake_y = torch.load(
        f'{fprefix}.pt')
    assert len(fake) == len(fake_y)
    return (fake, fake_y)


def add_color_border(x, ratio=0.05, c=[0, 1, 0]):
    assert len(x.shape) == 3  # a single image
    assert x.shape[0] == 3  # C, H, W
    assert x.min() >= 0
    assert x.max() <= 1
    D = x.shape[1]
    B = int(D * ratio)

    def set_color(xp):
        xp[0] = c[0]
        xp[1] = c[1]
        xp[2] = c[2]
    set_color(x[:, :, :B])
    set_color(x[:, :, -B:])
    set_color(x[:, :B, :])
    set_color(x[:, -B:, :])
    return x


def main(args):
    # Logging Prep
    os.makedirs(f'results/stats/evaluate_samples/nclass{args.nclass}', exist_ok=True)
    if args.name != 'load_samples_pt':
        fname = args.name
    else:
        fname = os.path.split(args.samples_pt_prefix)[1]
    if args.save_prefix:
        fname = args.save_prefix + '-' + fname

    results = {}
    # Load Data
    target_x, target_y = _load_real_cache()

    # Load Samples
    fake, fake_y = args.f_load()

    # FID
    # - select on classes where fake_y is available
    selected_x = []
    for y in fake_y.unique():
        selected_x.append(target_x[target_y == y])
    selected_x = torch.cat(selected_x)
    fid = run_fid(selected_x, fake)
    results['fid'] = fid

    # PRCD for all 100 ids
    prcd_results = defaultdict(list)
    for id in tqdm(range(args.nclass if not args.db else 2), desc='prcd loop'):
        # Maybe Skip
        if (fake_y == id).float().sum() == 0:
            continue

        prcd_runner = PRCD(
            lambda x: run_feature_extractor(postprocess(x.cuda())),
            target_x[target_y == id]
        )
        fake_c = fake[fake_y == id]
        D = prcd_runner.evaluate(fake_c)
        for k in D:
            prcd_results[k].append(D[k])
    df = pandas.DataFrame(prcd_results)
    df.to_csv(f'results/stats/evaluate_samples/nclass{args.nclass}/{fname}-prcd.csv')

    for k in prcd_results:
        results[k] = np.mean(prcd_results[k])

    # Evaluation Accuracy
    evaluation_classifier = PretrainedInsightFaceClassifier(
        'cuda:0', pad=bool(args.eval_cls_pad))

    acc_results = {}
    top5_acc_results = {}
    for id in tqdm(range(args.nclass if not args.db else 2), desc='acc loop'):
        # Maybe Skip
        if (fake_y == id).float().sum() == 0:
            continue

        x = fake[fake_y == id]
        if len(x) == 0:
            continue
        logits = evaluation_classifier.logits(x[:, [2, 1, 0]])
        # Top1
        preds = logits.max(1)[1]
        corrects = preds.cpu() == id
        acc = corrects.float().mean().item()
        acc_results[id] = acc

        # Top5
        top5 = torch.topk(logits, k=5, dim=1)[1]
        top5_corret = np.array([id in t for t in top5.cpu().numpy()])
        top5_acc = top5_corret.mean()
        top5_acc_results[id] = top5_acc
    avg_acc = np.mean(list(acc_results.values()))
    results['acc'] = avg_acc
    avg_top5_acc = np.mean(list(top5_acc_results.values()))
    results['top5_acc'] = avg_top5_acc
    print(avg_acc, avg_top5_acc)

    # Save
    df = pandas.DataFrame({fname: results})
    df.to_csv(f'results/stats/evaluate_samples/nclass{args.nclass}/{fname}.csv')
    acc_df = pandas.DataFrame({fname: acc_results})
    acc_df.to_csv(f'results/stats/evaluate_samples/nclass{args.nclass}/{fname}-accs.csv')
    acc_df = pandas.DataFrame({fname: top5_acc_results})
    acc_df.to_csv(f'results/stats/evaluate_samples/nclass{args.nclass}/{fname}-t5accs.csv')


def compute_entropy(p, epsilon=1e-4):
    p = p * (1 - epsilon) + .5 * epsilon
    return - p * torch.log(p)


def compute_kl(p, q, epsilon=1e-4):
    # Avoid 0
    p = p * (1 - epsilon) + .5 * epsilon
    q = q * (1 - epsilon) + .5 * epsilon
    return torch.mean(p * (torch.log(p) - torch.log(q)))


def main_plot(args):
    if args.name != 'load_samples_pt':
        fname = args.name
    else:
        fname = os.path.split(args.samples_pt_prefix)[1]
    if args.save_prefix:
        fname = args.save_prefix + '-' + fname
    os.makedirs(f'results/eval-sample-viz/{fname}', exist_ok=True)
    # Load Samples
    fake, fake_y = args.f_load()
    for id_start in range(0, args.nclass, args.every_nclass):
        # Aggregate
        ims = []
        for id in range(id_start, id_start + args.every_nclass):
            mask = fake_y == id
            if mask.float().sum() == 0:  # Blank Image Placeholder
                ims.append(torch.zeros(args.nperclass,
                                       3, 64, 64).to(fake.device))
            else:
                ims.append(fake[mask][:args.nperclass])
        ims = torch.cat(ims)

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(
            args.nperclass, args.every_nclass))
        imgrid = vutils.make_grid(
            ims, nrow=args.nperclass, padding=2, pad_value=0, normalize=True)
        imgrid = imgrid.cpu().numpy()
        im = np.transpose(imgrid, (1, 2, 0))
        ax.imshow(im, interpolation='bilinear')
        # Style
        plt.xticks([])
        plt.yticks([])
        impath = f"results/eval-sample-viz/{fname}/id{id_start}-{id_start+args.every_nclass}"
        plt.savefig(impath, bbox_inches='tight')


if __name__ == '__main__':
    # import sys
    # dev()
    # sys.exit(0)

    all_sample_choices = {
        'real': _load_real_cache,
        'load_samples_pt': _load_samples_pt,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True,
                        choices=list(all_sample_choices.keys()))
    parser.add_argument('--eval_what', type=str,
                        default='stats', choices=['stats', 'plot'])
    parser.add_argument('--samples_pt_prefix', type=str)
    parser.add_argument('--nclass', type=int, default=100)
    parser.add_argument('--nperclass', type=int, default=5)
    parser.add_argument('--every_nclass', type=int, default=10)
    parser.add_argument('--save_prefix', type=str, default='')
    parser.add_argument('--eval_cls_pad', type=int, default=0)
    parser.add_argument('--db', type=int, default=0)
    args = parser.parse_args()
    if args.name == 'load_samples_pt':
        args.f_load = lambda: _load_samples_pt(args, args.samples_pt_prefix)
    else:
        args.f_load = all_sample_choices[args.name]

    if args.eval_what == 'stats':
        main(args)
    elif args.eval_what == 'plot':
        main_plot(args)
    else:
        raise
