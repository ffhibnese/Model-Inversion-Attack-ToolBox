from .base import (
    BaseImageClassifier,
    BaseImageEncoder,
    HOOK_NAME_FEATURE,
    HOOK_NAME_HIDDEN,
)
from .wrappers import TorchvisionClassifierModel, VibWrapper, BiDOWrapper
from .classifier64 import (
    VGG16_64,
    IR152_64,
    FaceNet64,
    EfficientNet_b0_64,
    EfficientNet_b1_64,
    EfficientNet_b2_64,
)
from .classifier112 import FaceNet112
from .classifier_utils import generate_feature_statics
