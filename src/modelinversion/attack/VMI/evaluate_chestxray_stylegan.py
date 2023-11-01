import sys; sys.path.append('../stylegan2-ada-pytorch')
import legacy
import dnnlib
import os
import torch
import argparse
import torchvision.utils as vutils
from evaluate_samples_chestxray import load_class_balanced_real_data
from fid import run_fid
device = 'cuda:0'


network = '/scratch/hdd001/home/wangkuan/stylegan/run_scripts/May12-train-gan-chestxray.sh-1/auto/00003-chestxray-aux-auto2-resumefromprev/network-snapshot-000121.pkl'

# StyleGAN
print('Loading networks from "%s"...' % network)
with dnnlib.util.open_url(network) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
noise_mode = 'const'

with torch.no_grad():
    z_nuisance = torch.randn(50, G.z_dim).to(device).double()
    w_nuisance = G.mapping(z_nuisance, None)
    x = G.synthesis(w_nuisance, noise_mode=noise_mode)
vutils.save_image(x, 'cdb.jpeg', normalize=True)



# FID
# Load Data
target_x, _ = load_class_balanced_real_data()
target_x = target_x.repeat(1, 3, 1, 1)

# Samples
fake = []
for _ in range(30):
    with torch.no_grad():
        z_nuisance = torch.randn(100, G.z_dim).to(device).double()
        w_nuisance = G.mapping(z_nuisance, None)
        x = G.synthesis(w_nuisance, noise_mode=noise_mode)
    fake.append(x.cpu())
fake = torch.cat(fake)
fake = fake.repeat(1, 3, 1, 1)

# FID
fid = run_fid(target_x, fake)
print(f'FID: {fid}')