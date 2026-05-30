import sys
import os
import torch
import torch.nn as nn
sys.path.append('../../../src')

from modelinversion.models import TorchvisionClassifierModel, SimpleGenerator64
from modelinversion.datasets import CelebA224
from modelinversion.utils import Logger
import torchvision.transforms as T

class AutoEncoder(nn.Module):

    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)[0]
        x_hat = self.decoder(z)
        return x_hat

def main(dimension=256):

    save_dir = f'./results_low/{dimension}'

    logger = Logger(save_dir, 'train.log')

    device = 'cuda'

    encoder = TorchvisionClassifierModel('resnet50', num_classes=dimension, resolution=64, weights='DEFAULT')

    decoder = SimpleGenerator64(dimension)

    model = AutoEncoder(encoder, decoder).to(device)
    model.train()

    train_dataset = CelebA224('/data/<usrname>/datasets/pre_celeba_high/private_train', output_transform=T.Compose([
        T.ToTensor(),
        T.RandomResizedCrop(
            size=(224, 224), scale=(0.85, 1), ratio=(1, 1), antialias=True
        ),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        T.Resize((64, 64)),
        T.RandomHorizontalFlip(p=0.5),
    ]))

    train_dataset_noaug = CelebA224('/data/<usrname>/datasets/pre_celeba_high/private_train', output_transform=T.Compose([
        T.ToTensor(),
        T.Resize((64, 64))
    ]))

    test_dataset = CelebA224('/data/<usrname>/datasets/pre_celeba_high/private_test', output_transform=T.Compose([
        T.ToTensor(),
        T.Resize((64, 64))
    ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8)
    train_loader_noaug = torch.utils.data.DataLoader(train_dataset_noaug, batch_size=128, shuffle=False, num_workers=8)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75, 90], gamma=0.1)

    loss_fn = nn.MSELoss()

    from tqdm import tqdm

    for epoch in range(100):

        bar = tqdm(train_loader, leave=False, disable=dimension != 300)
        for i, (x, _) in enumerate(bar):

            x = x.to(device)
            x_hat = model(x)
            loss = loss_fn(x, x_hat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bar.set_description(f'Epoch: {epoch} Step: {i}, Loss: {loss.item():.4f}')

        lr_scheduler.step()

        if epoch % 5 == 0:

            test_losses = []
            
            with torch.no_grad():
                for i, (x, _) in enumerate(test_loader):
                    x = x.to(device)
                    x_hat = model(x)
                    loss = loss_fn(x, x_hat).cpu().item()
                    test_losses.append(loss)

            # print(f'Epoch: {epoch}, Test Loss: {torch.tensor(losses).mean())
            test_loss = torch.tensor(test_losses).mean().item()

            train_losses = []
            
            with torch.no_grad():
                for i, (x, _) in enumerate(train_loader_noaug):
                    x = x.to(device)
                    x_hat = model(x)
                    loss = loss_fn(x, x_hat).cpu().item()
                    train_losses.append(loss)

            # print(f'Epoch: {epoch}, Test Loss: {torch.tensor(losses).mean())
            train_loss = torch.tensor(train_losses).mean().item()

            logger.write(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

            # print(loss.item())


    encoder.save_pretrained(os.path.join(save_dir, 'encoder'))
    decoder.save_pretrained(os.path.join(save_dir, 'decoder'))
    logger.close()

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    dimensions = [2, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250, 300]
    # 
    for i in range(len(dimensions)-1):
        if os.fork() == 0:
            main(dimensions[i])
            exit(0)
    main(dimensions[-1])

