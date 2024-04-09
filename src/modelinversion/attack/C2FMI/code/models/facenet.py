import torch
from torch import nn
from .mobile_net import MobileNetV1
from .inception_resnetv1 import InceptionResnetV1
from torch.nn import functional as F
from .swin_transformer import SwinTransformer
from .MobileFaceNet import MobileFaceNet
from torchvision.transforms import Resize
from .model_irse import IR_50


class mobilenet_part(nn.Module):
    def __init__(self, pretrained_dir):
        super(mobilenet_part, self).__init__()
        self.model = MobileNetV1()
        if pretrained_dir:
            state_dict = torch.load(pretrained_dir)
            self.model.load_state_dict(state_dict, strict=False)

        del self.model.avgpool
        del self.model.fc

    def forward(self, x):
        x = self.model.stage1(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        return x


class inception_resnet_part(nn.Module):
    def __init__(self, pretrained_dir):
        super(inception_resnet_part, self).__init__()
        self.model = InceptionResnetV1()
        if pretrained_dir:
            state_dict = torch.load(pretrained_dir)
            self.model.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = self.model.conv2d_1a(x)
        x = self.model.conv2d_2a(x)
        x = self.model.conv2d_2b(x)
        x = self.model.maxpool_3a(x)
        x = self.model.conv2d_3b(x)
        x = self.model.conv2d_4a(x)
        x = self.model.conv2d_4b(x)
        x = self.model.repeat_1(x)
        x = self.model.mixed_6a(x)
        x = self.model.repeat_2(x)
        x = self.model.mixed_7a(x)
        x = self.model.repeat_3(x)
        x = self.model.block8(x)
        return x


class mobile_facenet_part(nn.Module):
    def __init__(self, pretrained_dir):
        super(mobile_facenet_part, self).__init__()
        self.model = MobileFaceNet()
        self.resize = Resize([112, 112])
        if pretrained_dir:
            state_dict = torch.load(pretrained_dir)
            self.model.load_state_dict(state_dict, strict=False)

        # del self.model.conv_6_dw
        del self.model.conv_6_flatten
        del self.model.linear
        del self.model.bn

    def forward(self, x):
        x = self.resize(x)  # B*3*112*112
        out = self.model.conv1(x)
        out = self.model.conv2_dw(out)
        out = self.model.conv_23(out)
        out = self.model.conv_3(out)
        out = self.model.conv_34(out)
        out = self.model.conv_4(out)
        out = self.model.conv_45(out)
        out = self.model.conv_5(out)
        out = self.model.conv_6_sep(out)
        out = self.model.conv_6_dw(out)  # B*512*7*7
        return out


class IR_50_part(nn.Module):
    def __init__(self, pretrained_dir):
        super(IR_50_part, self).__init__()
        self.model = IR_50([112, 112])
        self.resize = Resize([112, 112])
        if pretrained_dir:
            state_dict = torch.load(pretrained_dir)
            self.model.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = self.resize(x)
        x = self.model(x)
        x = x.unsqueeze(2)
        x = x.unsqueeze(3)
        return x


class Facenet(nn.Module):
    def __init__(
        self,
        backbone='mobile_net',
        dropout_keep_prob=0.5,
        embedding_size=128,
        num_classes=None,
        mode='train',
        pretrained_dir=None,
    ):
        super(Facenet, self).__init__()
        if backbone == 'mobile_net':
            self.backbone_name = 'MobileNet'
            self.backbone = mobilenet_part(pretrained_dir)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            flat_shape = 1024
        elif backbone == 'inception_resnetv1':
            self.backbone_name = 'InceptionResnetV1'
            self.backbone = inception_resnet_part(pretrained_dir)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            flat_shape = 1792
        elif backbone == 'swin_t':
            self.backbone_name = 'SwinTransformer_t'
            self.backbone = SwinTransformer(
                img_size=160, window_size=5, drop_rate=0.3, attn_drop_rate=0.15
            )
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            flat_shape = 768
        elif backbone == 'IR50':
            self.backbone_name = 'IR50'
            self.backbone = IR_50_part(pretrained_dir)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            flat_shape = 512
        elif backbone == 'mobile_facenet':
            self.backbone_name = 'MobileFaceNet'
            self.backbone = mobile_facenet_part(pretrained_dir)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            flat_shape = 512
            embedding_size = 512
            dropout_keep_prob = 0.9
        else:
            raise ValueError(
                f'Unsupported backbone `{backbone}`, Please use `mobile_net` or `inception_resnetv1`'
            )

        self.dropout = nn.Dropout(1 - dropout_keep_prob)
        self.bottleneck = nn.Linear(flat_shape, embedding_size, bias=False)
        self.final_bn = nn.BatchNorm1d(
            embedding_size, eps=0.001, momentum=0.1, affine=True
        )

        if mode == 'train':
            self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        # 返回归一化的feature: batch_size * embedding_size
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.bottleneck(x)
        x = self.final_bn(x)
        x = F.normalize(x, 2, dim=1)
        return x

    def forward_feature(self, x):
        # 返回未归一化和归一化的feature: batch_size * embedding_size
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.bottleneck(x)
        x_ = self.final_bn(x)
        x = F.normalize(x_, 2, dim=1)
        return x_, x

    def forward_classifier(self, x):
        # x为forward_feature计算得到的x_
        x = self.classifier(x)
        return x
