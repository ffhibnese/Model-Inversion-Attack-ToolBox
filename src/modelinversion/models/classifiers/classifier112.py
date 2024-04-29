from copy import deepcopy

from torch import Tensor
import torchvision

from ..base import ModelMixin
from ...utils import BaseHook

from .base import *
from .evolve import evolve


@register_model(name='facenet112')
class FaceNet112(BaseImageClassifier):

    @ModelMixin.register_to_config_init
    def __init__(
        self,
        num_classes=1000,
        register_last_feature_hook=False,
        backbone_path: Optional[str] = None,
    ):
        super(FaceNet112, self).__init__(
            112, 512, num_classes, register_last_feature_hook
        )
        self.feature = evolve.IR_50_112((112, 112))
        if backbone_path is not None:
            state_dict = torch.load(backbone_path, map_location='cpu')
            self.feature.load_state_dict(state_dict)
        self.feat_dim = 512

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

        self.feature_hook = FirstInputHook(self.fc_layer)

    def get_last_feature_hook(self) -> BaseHook:
        return self.feature_hook

    def preprocess_config_before_save(self, config):
        config = deepcopy(config)
        del config['backbone_path']
        return super().preprocess_config_before_save(config)

    def _forward_impl(self, image: Tensor, *args, **kwargs):
        feat = self.feature(image)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        return out
