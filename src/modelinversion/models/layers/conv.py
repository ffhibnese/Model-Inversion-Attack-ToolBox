import torch
from torch import nn
from torch.nn import functional as F


# class ComposeConv2D(nn.Module):

#     def __init__(self, conv_a, conv_b):
#         super(ComposeConv2D, self).__init__()
#         self.conv_a = conv_a
#         self.conv_b = conv_b

#     def forward(self, x):
#         x = self.conv_a(x)
#         x = self.conv_b(x)
#         return x


class Attention2D(nn.Module):
    def __init__(
        self,
        in_planes,
        K,
    ):
        super(Attention2D, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, K, 1)
        self.fc2 = nn.Conv2d(K, K, 1)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x, 1)


class DynamicConv2D(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        K=4,
    ):
        super(DynamicConv2D, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = Attention2D(in_planes, K)

        self.weight = nn.Parameter(
            torch.Tensor(K, out_planes, in_planes // groups, *kernel_size),
            requires_grad=True,
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None

    def forward(
        self, x
    ):  # 将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)  # 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(
            batch_size * self.out_planes, -1, *self.kernel_size
        )
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(
                x,
                weight=aggregate_weight,
                bias=aggregate_bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * batch_size,
            )
        else:
            output = F.conv2d(
                x,
                weight=aggregate_weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * batch_size,
            )

        output = output.view(
            batch_size, self.out_planes, output.size(-2), output.size(-1)
        )
        return output


if __name__ == '__main__':
    conv = DynamicConv2D(7, 11, 3, padding=1, K=5)
    t = torch.randn(17, 7, 23, 23)
    print(conv(t).shape)
