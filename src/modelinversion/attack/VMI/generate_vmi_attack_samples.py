import sys; sys.path.append('../stylegan2-ada-pytorch')
from attack_stylegan import ReparameterizedMVN, MineGAN, num_range, MixtureOfRMVN, LayeredMineGAN, LayeredFlowMiner, FlowMiner
import legacy
import dnnlib
import os
import torch
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--layers', type=str, default='0-9')
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--dry_run', action="store_true")
args = parser.parse_args()
args.lambda_trunc = 1
args.lambda_trunc_nuisance = 1


device = 'cuda:0'


output_dir = 'results/stylegan-attack-flow'
os.makedirs(output_dir, exist_ok=True)

epoch = args.epoch
method = 'layeredflow'
exp_path_without_id = os.path.join('results/celeba')
network = 'pretrained/stylegan/neurips2021-celeba/network-snapshot-002298.pkl'

# Check all files exist
if args.dry_run:
    passed = True
    for id in range(100):
        exp_dir = exp_path_without_id + f'-id{id}'
        if os.path.exists(os.path.join(exp_dir, f'miner_{epoch}.pt')):
            continue
        else:
            passed = False
            print(f"Dry run failed -- missing id:{id}")
    if passed:
        print("Dry run passed!!!")
    else:
        print("Dry run FAILED...")
    sys.exit(0)

# StyleGAN
print('Loading networks from "%s"...' % network)
with dnnlib.util.open_url(network) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
noise_mode = 'const'
l_identity = num_range(args.layers)
identity_mask = torch.zeros(1, G.mapping.num_ws, 1).to(device)
identity_mask[:, l_identity, :] = 1


all_imgs = []
ys = []
for id in range(0, 100):
    exp_dir = exp_path_without_id + f'-id{id}'

    if not os.path.exists(os.path.join(exp_dir, f'miner_{epoch}.pt')):
        continue

    if method == 'minegan':
        miner = ReparameterizedMVN(G.mapping.z_dim).to(device).double()
        minegan_Gmapping = MineGAN(miner, G.mapping)
    elif method == 'layeredminegan':
        miner = MixtureOfRMVN(512, 10).to(device).double()
        minegan_Gmapping = LayeredMineGAN(miner, G.mapping)
    elif method == 'flow':
        miner = FlowMiner(G.mapping.z_dim, 'shuffle', 50).to(device).double()
        minegan_Gmapping = MineGAN(miner, G.mapping)
    elif method == 'layeredflow':
        miner = LayeredFlowMiner(
            G.mapping.z_dim, G.mapping.num_ws, 'shuffle', 50).to(device).double()
        miner.eval()
        minegan_Gmapping = LayeredMineGAN(miner, G.mapping)

    miner_sd = torch.load(os.path.join(exp_dir, f'miner_{epoch}.pt'))
    miner.load_state_dict(miner_sd)

    def sample(z_nuisance, z_identity):
        w_nuisance = G.mapping(z_nuisance, None)
        w_identity = minegan_Gmapping(z_identity)
        w = (1 - identity_mask) * w_nuisance + identity_mask * w_identity
        x = G.synthesis(w, noise_mode=noise_mode)
        return x

    with torch.no_grad():
        z_nu = torch.randn(50, G.z_dim).to(device).double()
        z_id = torch.randn(50, G.z_dim).to(device).double()
        fakes = sample(z_nu, z_id)

    all_imgs.append(fakes)
    ys.append(id * torch.ones(50))
all_imgs = torch.cat(all_imgs)
ys = torch.cat(ys)

fname = 'stylegan-attack-with-labels-id0-100'

# save images
torch.save((all_imgs.clamp(-1, 1).detach().cpu(), ys), os.path.join('results/images_pt/', f'{fname}.pt'))
