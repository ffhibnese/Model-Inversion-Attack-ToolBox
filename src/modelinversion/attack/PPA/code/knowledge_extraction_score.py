import argparse

import torch
import torchvision.transforms as T
from rtpt import RTPT
from models.classifier import Classifier
from metrics.accuracy import Accuracy

from tqdm import tqdm

from datasets.attack_latents import AttackLatents
from datasets.custom_subset import ClassSubset
from datasets.facescrub import FaceScrub
from datasets.celeba import CelebA1000
from utils.stylegan import load_generator
import wandb


def main():

    parser = argparse.ArgumentParser(
        description='Compute information extraction score')
    parser.add_argument('-r',
                        '--runpath',
                        type=str,
                        dest="runpath",
                        help='Runpath of attack run')
    parser.add_argument('-e',
                        '--epochs',
                        type=int,
                        default=75,
                        dest="epochs",
                        help='Number of training epochs')
    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        default=32,
                        dest="batch_size",
                        help='Training batch size')
    parser.add_argument('-g',
                        '--generator',
                        type=str,
                        default='stylegan2-ada-pytorch/ffhq.pkl',
                        dest="generator",
                        help='StyleGAN2 generator weights')
    parser.add_argument('-d',
                        '--dataset',
                        type=str,
                        dest="dataset",
                        help='FaceScrub or CelebA')
    parser.add_argument('-u',
                        '--user',
                        type=str,
                        default='XX',
                        dest="user",
                        help='User name or initials')

    args = parser.parse_args()

    # Load attack run from WandB
    api = wandb.Api()
    run = api.run(args.runpath)

    torch.manual_seed(run.config['Seed'])

    # Set up RTPT
    rtpt = RTPT(args.user, 'Knowledge Extraction Score', args.epochs)
    rtpt.start()

    # Load StyleGAN2 generator
    generator = load_generator(args.generator).cuda()

    # Define dataset for attack results
    latent_transforms = T.Lambda(lambda x: x.repeat_interleave(18, 0).cuda())
    dataset = AttackLatents(args.runpath, transform=latent_transforms)
    latent_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True)
    
    # Define training and test augmentations
    transforms_train = T.Compose([
        T.CenterCrop(800),
        T.Resize((224, 224), antialias=True),
        T.RandomResizedCrop((224, 224),
                            scale=(0.8, 1.0),
                            ratio=(1.0, 1.0),
                            antialias=True),
        T.RandomHorizontalFlip()
    ])

    transforms_test = T.Compose([
        T.Resize((224, 224), antialias=True),
        T.ToTensor(),
        T.Normalize(0.5, 0.5)
    ])

    # Load training dataset of target model
    if args.dataset.lower().strip() == 'facescrub':
        train_set = FaceScrub('all',
                              train=True,
                              split_seed=run.config['Seed'],
                              transform=transforms_test)
    elif args.dataset.lower().strip() == 'celeba':
        train_set = CelebA1000(train=True,
                               split_seed=run.config['Seed'],
                               transform=transforms_test)
    else:
        raise ValueError('Invalid dataset specified')

    # Load test dataset of target model
    if args.dataset.lower().strip() == 'facescrub':
        test_set = FaceScrub('all',
                             train=False,
                             split_seed=run.config['Seed'],
                             transform=transforms_test)
    elif args.dataset.lower().strip() == 'celeba':
        test_set = CelebA1000(train=False,
                              split_seed=run.config['Seed'],
                              transform=transforms_test)
    else:
        raise ValueError('Invalid dataset specified')

    print(f'Evaluating on {test_set.name}')
    
    # Create dataloaders
    if dataset.target_identities != 'all':
        test_set = ClassSubset(test_set, dataset.target_identities)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=args.batch_size * 2,
                                              shuffle=False,
                                              num_workers=8)


    if dataset.target_identities != 'all':
        train_set = ClassSubset(train_set, dataset.target_identities)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size*2,
                                               shuffle=False,
                                               num_workers=8)

    # Define student model
    model = Classifier(num_classes=dataset.num_classes,
                       architecture='resnet50',
                       pretrained=True,
                       name='ResNet50')
    if torch.__version__.startswith('2.'):
        print('Compiling model with torch.compile')
        model.model = torch.compile(model.model)
    model = model.cuda()

    # Define optimizer and lr scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=1e-1,
                                weight_decay=1e-4,
                                momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=args.epochs,
                                                           eta_min=1e-4)

    # Training loop
    for i in range(args.epochs):
        print(f'Epoch {i+1}')
        model.train()
        num_corr, num_total = 0, 0
        for x, y in tqdm(latent_loader):
            with torch.no_grad():
                x, y = x.cuda(), y.cuda()
                x = generator.synthesis(x, noise_mode='const', force_fp32=True)
                x = transforms_train(x)
                x = torch.clamp(x, -1.0, 1.0)
            output = model(x)
            loss = torch.nn.functional.cross_entropy(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y_pred = output.argmax(dim=1)
            num_corr += (y_pred == y).sum()
            num_total += y.shape[0]
        print(f'Accuracy Synthetic: {num_corr/num_total:.4f}')
        scheduler.step()
        rtpt.step()

        acc, loss = model.evaluate(test_set,
                                   batch_size=2 * args.batch_size,
                                   metric=Accuracy)
        print(f'Accuracy Real: {acc:.4f}')

    # Final evaluation on real data
    model.eval()
    num_corr, num_total = 0, 0
    for x, y in train_loader:
        with torch.no_grad():
            x, y = x.cuda(), y.cuda()
            output = model(x)
            y_pred = output.argmax(dim=1)
            num_corr += (y_pred == y).sum()
            num_total += y.shape[0]
    run.summary["extraction_score_train"] = num_corr / num_total
    print(f'Knowledge Extraction Score computed on the training data: {num_corr / num_total:.2f}')

    num_corr, num_total = 0, 0
    for x, y in test_loader:
        with torch.no_grad():
            x, y = x.cuda(), y.cuda()
            output = model(x)
            y_pred = output.argmax(dim=1)
            num_corr += (y_pred == y).sum()
            num_total += y.shape[0]
    rtpt.step()
    run.summary["extraction_score_test"] = num_corr / num_total
    print(f'Knowledge Extraction Score computed on the test data: {num_corr / num_total:.2f}')

    run.summary.update()


if __name__ == '__main__':
    main()