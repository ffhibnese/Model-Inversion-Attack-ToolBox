import torch
from torch import nn

def dconv_bn_relu(in_dim, out_dim):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
        nn.BatchNorm2d(out_dim),
        nn.ReLU())

class ContextNetwork(nn.Module):
    def __init__(self):
        super(ContextNetwork, self).__init__()
        # input_shape: (None, 4, img_h, img_w)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        # input_shape: (None, 32, img_h, img_w)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU()
        # input_shape: (None, 64, img_h//2, img_w//2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.act3 = nn.ReLU()
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.act4 = nn.ReLU()
        # input_shape: (None, 128, img_h//4, img_w//4)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.act5 = nn.ReLU()
        # input_shape: (None, 128, img_h//4, img_w//4)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.act6 = nn.ReLU()
        # input_shape: (None, 128, img_h//4, img_w//4)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, dilation=2, padding=2)
        self.bn7 = nn.BatchNorm2d(128)
        self.act7 = nn.ReLU()
        # input_shape: (None, 128, img_h//4, img_w//4)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, dilation=4, padding=4)
        self.bn8 = nn.BatchNorm2d(128)
        self.act8 = nn.ReLU()
        # input_shape: (None, 128, img_h//4, img_w//4)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, stride=1, dilation=8, padding=8)
        self.bn9 = nn.BatchNorm2d(128)
        self.act9 = nn.ReLU()
        # input_shape: (None, 128, img_h//4, img_w//4)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=3, stride=1, dilation=16, padding=16)
        self.bn10 = nn.BatchNorm2d(128)
        self.act10 = nn.ReLU()
        
        

    def forward(self, x):
        x = self.bn1(self.act1(self.conv1(x)))
        x = self.bn2(self.act2(self.conv2(x)))
        x = self.bn3(self.act3(self.conv3(x)))
        x = self.bn4(self.act4(self.conv4(x)))
        x = self.bn5(self.act5(self.conv5(x)))
        x = self.bn6(self.act6(self.conv6(x)))
        x = self.bn7(self.act7(self.conv7(x)))
        x = self.bn8(self.act8(self.conv8(x)))
        x = self.bn9(self.act9(self.conv9(x)))
        x = self.bn10(self.act10(self.conv10(x)))
        return x

class IdentityGenerator(nn.Module):

    def __init__(self, in_dim = 100, dim=64):
        super(IdentityGenerator, self).__init__()

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2))

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y

class InversionNet(nn.Module):
    def __init__(self, out_dim = 128):
        super(InversionNet, self).__init__()
        
        # input [4, h, w]  output [256, h // 4, w // 4]
        self.ContextNetwork = ContextNetwork()
        # input [z_dim] output[128, 16, 16]
        self.IdentityGenerator = IdentityGenerator()

        self.dim = 128 + 128
        self.out_dim = out_dim
        
        self.Dconv = nn.Sequential(
            dconv_bn_relu(self.dim, self.out_dim),
            dconv_bn_relu(self.out_dim, self.out_dim // 2))

        self.Conv = nn.Sequential(
            nn.Conv2d(self.out_dim // 2, self.out_dim // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim // 4),
            nn.ReLU(),
            nn.Conv2d(self.out_dim // 4, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid())


    def forward(self, inp):
        # x.shape [4, h, w]  z.shape [100]
        x, z = inp
        context_info = self.ContextNetwork(x)
        identity_info = self.IdentityGenerator(z)
        # []
        y = torch.cat((context_info, identity_info), dim=1)
        y = self.Dconv(y)
        y = self.Conv(y)

        return y