from .base import (
    BaseImageClassifier,
    BaseImageEncoder,
    HOOK_NAME_FEATURE,
    HOOK_NAME_HIDDEN,
)
from .wrappers import TorchvisionClassifierModel, VibWrapper, BiDOWrapper
from .classifier64 import VGG16_64, IR152_64, FaceNet64
from .classifier112 import FaceNet112
