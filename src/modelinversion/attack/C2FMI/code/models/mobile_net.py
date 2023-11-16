from torch import nn

def conv_bn(input_channels, output_channels, kernal_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernal_size, stride, padding, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU6()
    )

def conv_dw(input_channels, output_channels, kernal_size=3, stride=1, padding=1):
    # depthwise separable convolution
    return nn.Sequential(
        nn.Conv2d(input_channels, input_channels, kernal_size, stride, padding, groups=input_channels, bias=False),
        nn.BatchNorm2d(input_channels),
        nn.ReLU6(),

        nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU6()
    )

class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 32, stride=2),  # 3,160,160 -> 32,80,80  C*H*W
            conv_dw(32,64, stride=1),  # 32,80,80 -> 64,80,80

            conv_dw(64, 128, stride=2),   # 64,80,80 -> 128,40,40
            conv_dw(128, 128, stride=1),  # 128,40,40 -> 128,40,40

            conv_dw(128, 256, stride=2),  # 128,40,40 -> 256,20,20
            conv_dw(256, 256, stride=1),  # 256,20,20 -> 256,20,20
        )

        self.stage2 = nn.Sequential(
            # 256,20,20 -> 512,10,10
            conv_dw(256, 512, stride=2),
            conv_dw(512, 512, stride=1),
            conv_dw(512, 512, stride=1),
            conv_dw(512, 512, stride=1),
            conv_dw(512, 512, stride=1),
            conv_dw(512, 512, stride=1),
        )

        self.stage3 = nn.Sequential(
            # 512,10,10 -> 1024,5,5
            conv_dw(512, 1024, stride=2),
            conv_dw(1024, 1024, stride=1),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
