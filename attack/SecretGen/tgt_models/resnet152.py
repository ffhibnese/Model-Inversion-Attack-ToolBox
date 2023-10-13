import torch
import torch.nn as nn
import torchvision


class ResNet152(nn.Module):
    def __init__(self, num_classes = 1000, vis=False):
        super(ResNet152, self).__init__()
        self.vis = vis
        model = torchvision.models.resnet152(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.feat_dim = 2048
        self.num_of_classes = num_classes
        self.fc_layer = nn.Sequential(
            nn.Linear(self.feat_dim, self.num_of_classes),
            nn.Softmax(dim = 1))

    def classifier(self, x):
        out = self.fc_layer(x)
        __, iden = torch.max(out, dim = 1)
        return out, iden

    def forward(self, x):
        if self.vis:
            out = []
            for module in self.feature[0]:
                x = module(x)
                out.append(torch.flatten(x, 1))
            x = x.contiguous().view(x.size(0), -1)
            for module in self.fc_layer:
                x = module(x)
                out.append(torch.flatten(x, 1))
            return out
                
        feature = self.feature(x)
        feature = feature.contiguous().view(feature.size(0), -1)
        out, iden = self.classifier(feature)
        return feature, out, iden