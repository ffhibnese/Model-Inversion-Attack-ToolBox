import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from dataloader import CelebA
from tgt_models.vgg16 import VGG16
from tgt_models.resnet152 import ResNet152
from tensorboardX import SummaryWriter
import os.path as osp
import os
import numpy as np
import random


parser = argparse.ArgumentParser()
parser.add_argument('--name', '-n', required=True, type=str, choices=['vgg16', 'resnet152'], help='type of model to use')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--max_epoch', default=300, type=int, help='training epochs')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
opt = parser.parse_args()
print(opt)


torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if opt.name == 'vgg16':
    net = VGG16(num_classes=1000).to(device)
elif opt.name == 'resnet152':
    net = ResNet152(num_classes=1000).to(device)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)


trainset = CelebA(split='pri')
testset = CelebA(split='pri-dev')
trainloader = DataLoader(trainset, opt.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
testloader = DataLoader(testset, opt.batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)

nll_loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9)

max_epoch = opt.max_epoch
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)

writer = SummaryWriter(log_dir=osp.join('logs', f'{opt.name}-train-pri'))

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

        optimizer.zero_grad()
        _, logits, _ = net(inputs)
        loss = nll_loss(torch.log(logits), targets)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        total += targets.size(0)
        correct += len(preds[preds==targets])

        progress_bar.set_description(f'train loss: {loss:.4f}')

        writer.add_scalar('train loss', loss, train_step)
        train_step += 1

    acc = 100 * correct / total
    writer.add_scalar('train acc', acc, epoch)
    writer.add_scalar('lr', scheduler.get_lr()[0], epoch)



def test(epoch):
    global test_step
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(testloader)
        for inputs, _, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            _, logits, _  = net(inputs)
            loss = nll_loss(torch.log(logits), targets)

            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            total += targets.size(0)
            correct += len(preds[preds==targets])

            progress_bar.set_description(f'test loss: {loss:.4f}')

            writer.add_scalar('test loss', loss, test_step)
            test_step += 1
        
        acc = 100 * correct / total
        writer.add_scalar('test acc', acc, epoch)

    state = {
        'state_dict': net.state_dict(),
        'acc': acc,
    }
    if not osp.isdir('premodels'):
        os.mkdir('premodels')
    torch.save(state, f'./premodels/{opt.name}-pri.tar')


for epoch in range(max_epoch):
    train(epoch)
    test(epoch)
    scheduler.step()