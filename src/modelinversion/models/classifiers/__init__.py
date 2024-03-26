from .base import BaseImageClassifier, BaseImageEncoder, HOOK_NAME_FEATURE
from .wrappers import TorchvisionClassifierModel, VibWrapper
from .classifier64 import VGG16_64, IR152_64, FaceNet64
from .classifier112 import FaceNet112