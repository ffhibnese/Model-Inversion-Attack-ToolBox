import torch
from torch import nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class Conv_block(nn.Module):
    def __init__(
        self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1
    ):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(
            in_c,
            out_channels=out_c,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Linear_block(nn.Module):
    def __init__(
        self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1
    ):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(
            in_c,
            out_channels=out_c,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Depth_Wise(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        residual=False,
        kernel=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        groups=1,
    ):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(
            in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1)
        )
        self.conv_dw = Conv_block(
            groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride
        )
        self.project = Linear_block(
            groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1)
        )
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(nn.Module):
    def __init__(
        self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)
    ):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(
                Depth_Wise(
                    c,
                    c,
                    residual=True,
                    kernel=kernel,
                    padding=padding,
                    stride=stride,
                    groups=groups,
                )
            )
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class MobileFaceNet(nn.Module):
    def __init__(self, embedding_size=512):
        super(MobileFaceNet, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(
            64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64
        )
        self.conv_23 = Depth_Wise(
            64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128
        )
        self.conv_3 = Residual(
            64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv_34 = Depth_Wise(
            64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256
        )
        self.conv_4 = Residual(
            128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv_45 = Depth_Wise(
            128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512
        )
        self.conv_5 = Residual(
            128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv_6_sep = Conv_block(
            128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0)
        )
        self.conv_6_dw = Linear_block(
            512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0)
        )
        self.conv_6_flatten = Flatten()
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return l2_norm(out)


class MobileFaceNetPart(MobileFaceNet):
    def __init__(self, embedding_size=128):
        super(MobileFaceNetPart, self).__init__(embedding_size=embedding_size)

    def forward(self, x, resize_112=112):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        return out


if __name__ == '__main__':
    model = MobileFaceNetPart(embedding_size=128).cuda()
    model.eval()
    input = torch.randn((1, 3, 112, 112)).cuda()
    out = model(input)
    print(out.shape)
