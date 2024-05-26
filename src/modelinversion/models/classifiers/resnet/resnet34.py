import torch.nn as nn

class ResNet(nn.Module):
    maml = False  # Default

    def __init__(self, nc, block, list_of_num_layers, list_of_out_dims, flatten=True, final_fmap_size=7):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet, self).__init__()
        assert len(list_of_num_layers) == 4, 'Can have only four stages'
        if self.maml:
            conv1 = Conv2d_fw(nc, 64, kernel_size=7, stride=2, padding=3,
                              bias=False)
            bn1 = BatchNorm2d_fw(64)
        else:
            conv1 = nn.Conv2d(nc, 64, kernel_size=7, stride=2, padding=3,
                              bias=False)
            bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)

        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4):

            for j in range(list_of_num_layers[i]):
                half_res = (i >= 1) and (j == 0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(final_fmap_size)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [indim, final_fmap_size, final_fmap_size]

        self.trunk = nn.Sequential(*trunk)

    def forward(self, x):
        out = self.trunk(x)
        return out

def ResNetL_I(L, imgSize, nc, flatten=True):
    if imgSize == 32:
        ffs = 1
    elif imgSize == 64:
        ffs = 2
    elif imgSize == 128:
        ffs = 4
    elif imgSize == 256:
        ffs = 8
    else:
        raise

    if L == 10:
        net = ResNet(nc, SimpleBlock, [1, 1, 1, 1], [
                     64, 128, 256, 512], flatten, final_fmap_size=ffs)
    elif L == 34:
        net = ResNet(nc, SimpleBlock, [3, 4, 6, 3], [
                     64, 128, 256, 512], flatten, final_fmap_size=ffs)
    elif L == 50:
        net = ResNet(nc, BottleneckBlock, [3, 4, 6, 3], [
                     256, 512, 1024, 2048], flatten, final_fmap_size=ffs)
    return net

class ResNetCls1(nn.Module):
    def __init__(self, nc=3, zdim=2, imagesize=32, nclass=10, resnetl=10, dropout=0):
        super(ResNetCls1, self).__init__()
        self.backbone = ResNetL_I(resnetl, imagesize, nc)
        self.fc1 = nn.Linear(self.backbone.final_feat_dim, zdim)
        self.bn1 = nn.BatchNorm1d(self.backbone.final_feat_dim)
        self.fc2 = nn.Linear(zdim, nclass)
        self.bn2 = nn.BatchNorm1d(zdim)
        if dropout > 0:
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.dropout = dropout

    def embed_img(self, x):
        x = self.backbone(x)
        x = F.relu(x)
        if 'dropout' in dir(self) and self.dropout > 0:
            x = self.dropout1(x)
        x = self.bn1(x)
        x = self.fc1(x)
        return x

    def embed(self, x):
        return self.embed_img(x)

    def z_to_logits(self, z):
        z = F.relu(z)
        if 'dropout' in dir(self) and self.dropout > 0:
            z = self.dropout2(z)
        z = self.bn2(z)
        z = self.fc2(z)
        return z

    def logits(self, x):
        return self.z_to_logits(self.embed(x))

    def z_to_lsm(self, z):
        z = self.z_to_logits(z)
        return F.log_softmax(z, dim=1)

    def forward(self, x):
        x = self.embed_img(x)
        return self.z_to_lsm(x)