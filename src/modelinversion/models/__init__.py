from .modelresult import ModelResult
from .facenet.facenet import FaceNet, FaceNet64
from .ir152.ir152 import IR152
from .vgg.vgg16 import VGG16
from .vgg.vgg_face_dag import Vgg_face_dag
from .resnet.resnet50_scratch_dag import Resnet50_scratch_dag
from .efficientnet.efficientnet import EfficientNet_b0, EfficientNet_b1, EfficientNet_b2
from .inception.incv1 import InceptionResnetV1
from .get_models import get_model, NUM_CLASSES
from .base import BaseTargetModel