import torch
from tqdm import tqdm
from torch import nn

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def one_epoch_update(net, trip_loss, optimizer, epoch, epoch_step, epoch_step_val, train_loader,
                     val_loader, final_epoch, batch_size, using_gpu):
    total_triple_loss   = 0
    total_CE_loss       = 0
    total_accuracy      = 0

    val_total_triple_loss   = 0
    val_total_CE_loss       = 0
    val_total_accuracy      = 0

    net.train()
    print('Start Training')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{final_epoch}',postfix=dict,mininterval=0.3) as pbar:

        for iteration, batch in enumerate(train_loader):
            if iteration >= epoch_step:
                break
            images, labels = batch
            with torch.no_grad():
                if using_gpu:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    labels  = torch.from_numpy(labels).long().cuda()
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    labels  = torch.from_numpy(labels).long()

            optimizer.zero_grad()
            if using_gpu:
                before_norm, outputs1 = net.module.forward_feature(images)
                outputs2 = net.module.forward_classifier(before_norm)
            else:
                before_norm, outputs1 = net.forward_feature(images)
                outputs2 = net.forward_classifier(before_norm)

            _triplet_loss       = trip_loss(outputs1, batch_size)
            _CE_loss = nn.CrossEntropyLoss()(outputs2, labels)
            _loss               = _triplet_loss + _CE_loss

            _loss.backward()
            optimizer.step()

            with torch.no_grad():
                accuracy = torch.mean((torch.argmax(outputs2, dim=-1) == labels).type(torch.FloatTensor))

            total_triple_loss += _triplet_loss.item()
            total_CE_loss     += _CE_loss.item()
            total_accuracy    += accuracy.item()

            pbar.set_postfix(**{'total_triple_loss' : total_triple_loss / (iteration + 1),
                                'total_CE_loss'     : total_CE_loss / (iteration + 1),
                                'accuracy'          : total_accuracy / (iteration + 1),
                                'lr'                : get_lr(optimizer)})
            pbar.update(1)
    print('Finish Train')
    train_acc = total_accuracy / (iteration + 1)

    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val,desc=f'Epoch {epoch + 1}/{final_epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(val_loader):
            if iteration >= epoch_step_val:
                break
            images, labels = batch
            with torch.no_grad():
                if using_gpu:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    labels  = torch.from_numpy(labels).long().cuda()
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    labels  = torch.from_numpy(labels).long()

                if using_gpu:
                    before_norm, outputs1 = net.module.forward_feature(images)
                    outputs2 = net.module.forward_classifier(before_norm)
                else:
                    before_norm, outputs1 = net.forward_feature(images)
                    outputs2 = net.forward_classifier(before_norm)

                _triplet_loss       = trip_loss(outputs1, batch_size)
                _CE_loss = nn.CrossEntropyLoss()(outputs2, labels)
                _loss               = _triplet_loss + _CE_loss

                accuracy = torch.mean((torch.argmax(outputs2, dim=-1) == labels).type(torch.FloatTensor))

                val_total_triple_loss += _triplet_loss.item()
                val_total_CE_loss     += _CE_loss.item()
                val_total_accuracy    += accuracy.item()

            pbar.set_postfix(**{'total_triple_loss' : val_total_triple_loss / (iteration + 1),
                                'total_CE_loss'     : val_total_CE_loss / (iteration + 1),
                                'accuracy'          : val_total_accuracy / (iteration + 1),
                                'lr'                : get_lr(optimizer)})
            pbar.update(1)
    print('Finish Validation')
    val_acc = val_total_accuracy / (iteration + 1)

    total_loss = (total_triple_loss + total_CE_loss) / epoch_step
    total_val_loss = (val_total_triple_loss + val_total_CE_loss) / epoch_step_val

    print(f'Epoch: {epoch+1}/{final_epoch}')
    print(f'Total Loss: {total_loss}')
    if (epoch+1) % 2 == 0:
        if using_gpu:
            torch.save(net.module.state_dict(), f'facenet_logs/Backbone{net.module.backbone_name}-Epoch{epoch+1}-Train_Acc{round(train_acc,4)}-Val_Acc{round(val_acc,4)}.pth')
        else:
            torch.save(net.state_dict(), f'facenet_logs/Backbone{net.backbone_name}-Epoch{epoch+1}-Train_Acc{round(train_acc,4)}-Val_Acc{round(val_acc,4)}.pth')

    return total_val_loss
