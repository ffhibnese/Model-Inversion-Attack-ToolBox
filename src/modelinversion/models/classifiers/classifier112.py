from torch import Tensor
import torchvision

from ...utils import BaseHook

from .base import *
from .evolve import evolve


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class FaceNet112(BaseImageClassifier):
    def __init__(self, num_classes=1000, register_last_feature_hook=False):
        super(FaceNet112, self).__init__(112, 512, num_classes, register_last_feature_hook)
        self.feature = evolve.IR_50_112((112, 112))
        self.feat_dim = 512
        # self.num_classes = num_classes
        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
        
        self.feature_hook = FirstInputHook(self.fc_layer)
        
    def get_last_feature_hook(self) -> BaseHook:
        return self.feature_hook
    
    def _forward_impl(self, image: Tensor, *args, **kwargs):
        feat = self.feature(image)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        return out
        
    # def get_feature_dim(self):
    #     return 512
    
    # def create_hidden_hooks(self) -> list:
        
    #     hiddens_hooks = []
        
    #     length_hidden = len(self.feature.body)
        
    #     num_body_monitor = 4
    #     offset = length_hidden // num_body_monitor
    #     for i in range(num_body_monitor):
    #         hiddens_hooks.append(OutputHook(self.feature.body[offset * (i+1) - 1]))
        
    #     hiddens_hooks.append(OutputHook(self.feature.output_layer))
    #     return hiddens_hooks
    
    # def freeze_front_layers(self) -> None:
    #     length_hidden = len(self.feature.body)
    #     for i in range(int(length_hidden * 2 // 3)):
    #         self.feature.body[i].requires_grad_(False)

    # def predict(self, x):
    #     feat = self.feature(x)
    #     feat = feat.view(feat.size(0), -1)
    #     out = self.fc_layer(feat)
    #     return out

    # def forward(self, x):
    #     # print("input shape:", x.shape)
    #     # import pdb; pdb.set_trace()
        
    #     if x.shape[-1] != self.resolution or x.shape[-2] != self.resolution:
    #         x = resize(x, [self.resolution, self.resolution])

    #     feat = self.feature(x)
    #     feat = feat.view(feat.size(0), -1)
    #     out = self.fc_layer(feat)
    #     return ModelResult(out, [feat])