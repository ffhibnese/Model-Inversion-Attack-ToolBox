import torch
from torch.nn.modules.loss import _Loss

def completion_network_loss(input, output, mask):
    bs = input.size(0)
    loss = torch.sum(torch.abs(output * mask - input * mask)) / bs
    #return mse_loss(output * mask, input * mask)
    return loss

def noise_loss(V, img1, img2):
    feat1, __, ___ = V(img1)
    feat2, __, ___ = V(img2)
    
    loss = torch.mean(torch.abs(feat1 - feat2))
    #return mse_loss(output * mask, input * mask)
    return loss

class ContextLoss(_Loss):
    def forward(self, mask, gen, images):
        bs = gen.size(0)
        context_loss = torch.sum(torch.abs(torch.mul(mask, gen) - torch.mul(mask, images))) / bs
        return context_loss

class CrossEntropyLoss(_Loss):
    def forward(self, out, gt):
        bs = out.size(0)
        #print(out.size(), gt.size())
        loss = - torch.mul(gt.float(), torch.log(out.float() + 1e-7))
        loss = torch.sum(loss) / bs
        return loss

class FeatLoss(_Loss):
    def forward(self, fake_feat, real_feat):
        num = len(fake_feat)
        loss = torch.zeros(1).cuda()
        for i in range(num):
            loss += torch.mean(torch.abs(fake_feat[i] - real_feat[i]))
        
        return loss

