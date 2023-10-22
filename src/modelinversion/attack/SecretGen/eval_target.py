import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from dataloader import CelebAVirtual, CelebA, FaceScrub
# from tensorboardX import SummaryWriter
import os.path as osp
import os
import numpy as np
import random
from facenet import FaceNet152
from utils import low2high112


parser = argparse.ArgumentParser()
parser.add_argument('--name', default='ir152', type=str, help='evaluation model')
parser.add_argument('--batch_size', '-b', default=64, type=int, help='batch size')
parser.add_argument('--max_epoch', default=20, type=int, help='training epochs')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--test_data', required=True, type=str, help='data for evaluation')
opt = parser.parse_args()
print(opt)


torch.manual_seed(0)
torch.cuda.manual_seed(0)


eval_model = opt.name
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if eval_model == 'ir152':
    net = FaceNet152(num_classes=1000)
    net.feature.load_state_dict(torch.load('premodels/ir152.pth'))
    net = net.to(device)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)


trainset = CelebAVirtual(osp.join('data', opt.test_data), split='train')
devset = CelebAVirtual(osp.join('data', opt.test_data), split='dev')
testset = FaceScrub(split='pri-dev')

print(len(trainset))
print(len(devset))
print(len(testset))

trainloader = DataLoader(trainset, opt.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
devloader = DataLoader(devset, opt.batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
testloader = DataLoader(testset, opt.batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)

ce_loss = nn.CrossEntropyLoss()
nll_loss = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.max_epoch)
# writer = SummaryWriter(log_dir=osp.join('logs', f'eval_{opt.name}_{opt.train_data}_{opt.test_data}'))

train_step = 0
test_step = 0

def train(epoch):
    global train_step

    print('\nEpoch: %d' % epoch)
    net.train()
    correct = 0
    total = 0
    progress_bar = tqdm(trainloader)
    for inputs, _, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        if eval_model == 'ir152':
            inputs = low2high112(inputs)

        optimizer.zero_grad()
        _, out, iden = net(inputs)
        out = torch.log(out)
        loss = nll_loss(out, targets)
        loss.backward()
        optimizer.step()

        total += targets.size(0)
        correct += len(iden[iden==targets])

        progress_bar.set_description(f'train loss: {loss:.4f}')

        # writer.add_scalar('train loss', loss, train_step)
        train_step += 1

    acc = 100 * correct / total
    # writer.add_scalar('train acc', acc, epoch)
    # writer.add_scalar('lr', scheduler.get_lr()[0], epoch)


best_acc = 0

@torch.no_grad()
def dev(epoch):
    global test_step
    global best_acc
    net.eval()
    correct = 0
    total = 0

    progress_bar = tqdm(devloader)
    for inputs, _, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        if eval_model == 'ir152':
            inputs = low2high112(inputs)

        _, out, iden = net(inputs)
        out = torch.log(out)
        loss = nll_loss(out, targets)

        total += targets.size(0)
        correct += len(iden[iden==targets])

        progress_bar.set_description(f'test loss: {loss:.4f}')

        # writer.add_scalar('test loss', loss, test_step)
        test_step += 1
    
    acc = 100 * correct / total
    # writer.add_scalar('test acc', acc, epoch)

    state = {
        'state_dict': net.state_dict(),
        'acc': acc,
    }
    if acc > best_acc:
        best_acc = acc
        torch.save(state, f'./checkpoint/{opt.name}_{opt.test_data}.tar')
    

@torch.no_grad()
def test():
    net.load_state_dict(torch.load(f'./checkpoint/{opt.name}_{opt.test_data}.tar')['state_dict'])
    net.eval()
    correct = 0
    correct_top5 = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(testloader)
        for inputs, _, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            if eval_model == 'ir152':
                inputs = low2high112(inputs)

            _, pred_outputs, iden = net(inputs)
            total += targets.size(0)
            correct += len(iden[iden==targets])

            pred_class_top5 = torch.topk(pred_outputs, k=5, dim=-1).indices
            gt_class_idx = targets.unsqueeze(-1).repeat(1, 5)
            correct_top5 += int(torch.sum((torch.sum((gt_class_idx == pred_class_top5), dim=1) > 0), dim=0))
        
            acc = correct / total
            acc_top5 = correct_top5 / total
            progress_bar.set_description('test acc: {:.3f}, test acc top5: {:.3f}'.format(acc, acc_top5))


# os.makedirs('checkpoint', exist_ok=True)
# for epoch in range(opt.max_epoch):
#     train(epoch)
#     dev(epoch)
#     scheduler.step()

# print('dev acc: ', best_acc)
test()