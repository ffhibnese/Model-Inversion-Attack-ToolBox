import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from utils import LinearWeightNorm


class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims, mean=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        init.normal(self.T, 0, 1)

    def forward(self, x):
        # x is NxA
        # T is AxBxC
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        M = matrices.unsqueeze(0)  # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3)  # NxNxB
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1)  # NxB, subtract self distance
        if self.mean:
            o_b /= x.size(0) - 1

        x = torch.cat([x, o_b], 1)
        return x


class MinibatchDiscriminator(nn.Module):
    def __init__(self, in_dim=3, dim=64, n_classes=1000):
        super(MinibatchDiscriminator, self).__init__()
        self.n_classes = n_classes

        def conv_ln_lrelu(in_dim, out_dim, k, s, p):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, k, s, p),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.layer1 = conv_ln_lrelu(in_dim, dim, 5, 2, 2)
        self.layer2 = conv_ln_lrelu(dim, dim * 2, 5, 2, 2)
        self.layer3 = conv_ln_lrelu(dim * 2, dim * 4, 5, 2, 2)
        self.layer4 = conv_ln_lrelu(dim * 4, dim * 4, 3, 2, 1)
        self.mbd1 = MinibatchDiscrimination(dim * 4 * 4 * 4, 64, 50)
        self.fc_layer = nn.Linear(dim * 4 * 4 * 4 + 64, self.n_classes)

    def forward(self, x):
        out = []
        bs = x.shape[0]
        feat1 = self.layer1(x)
        out.append(feat1)
        feat2 = self.layer2(feat1)
        out.append(feat2)
        feat3 = self.layer3(feat2)
        out.append(feat3)
        feat4 = self.layer4(feat3)
        out.append(feat4)
        feat = feat4.view(bs, -1)
        # print('feat:', feat.shape)
        mb_out = self.mbd1(feat)  # Nx(A+B)
        y = self.fc_layer(mb_out)

        return feat, y
        # return mb_out, y


class Discriminator(nn.Module):
    def __init__(self, in_dim=3, dim=64, n_classes=1000):
        super(Discriminator, self).__init__()
        self.n_classes = n_classes

        def conv_ln_lrelu(in_dim, out_dim, k, s, p):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, k, s, p),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.layer1 = conv_ln_lrelu(in_dim, dim, 5, 2, 2)
        self.layer2 = conv_ln_lrelu(dim, dim * 2, 5, 2, 2)
        self.layer3 = conv_ln_lrelu(dim * 2, dim * 4, 5, 2, 2)
        self.layer4 = conv_ln_lrelu(dim * 4, dim * 4, 3, 2, 1)
        self.fc_layer = nn.Linear(dim * 4 * 4 * 4, self.n_classes)

    def forward(self, x):
        bs = x.shape[0]
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        feat = feat4.view(bs, -1)
        y = self.fc_layer(feat)

        return feat, y


class DiscriminatorMNIST(nn.Module):
    def __init__(self, d_input_dim=1024):
        super(DiscriminatorMNIST, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward method
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        y = self.fc4(x)
        y = y.view(-1)

        return y


class DGWGAN32(nn.Module):
    def __init__(self, in_dim=1, dim=64):
        super(DGWGAN32, self).__init__()

        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.layer1 = nn.Sequential(nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2))
        self.layer2 = conv_ln_lrelu(dim, dim * 2)
        self.layer3 = conv_ln_lrelu(dim * 2, dim * 4)
        self.layer4 = nn.Conv2d(dim * 4, 1, 4)

    def forward(self, x):
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        y = self.layer4(feat3)
        y = y.view(-1)
        return y


class DGWGAN(nn.Module):
    def __init__(self, in_dim=3, dim=64):
        super(DGWGAN, self).__init__()

        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_ln_lrelu(dim, dim * 2),
            conv_ln_lrelu(dim * 2, dim * 4),
            conv_ln_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 4))

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y


class DLWGAN(nn.Module):
    def __init__(self, in_dim=3, dim=64):
        super(DLWGAN, self).__init__()

        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.layer1 = nn.Sequential(nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2))
        self.layer2 = conv_ln_lrelu(dim, dim * 2)
        self.layer3 = conv_ln_lrelu(dim * 2, dim * 4)
        self.layer4 = nn.Conv2d(dim * 4, 1, 4)

    def forward(self, x):
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        y = self.layer4(feat3)
        return y


class MyDiscriminator(nn.Module):
    def __init__(self, in_features=2048):
        super(MyDiscriminator, self).__init__()
        model = [
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        output = self.model(x)
        return output.squeeze(1)


class MyDiscriminator2(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(MyDiscriminator2, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        # if input.is_cuda and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)
