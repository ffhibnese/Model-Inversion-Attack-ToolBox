from .modelresult import ModelResult
from .facenet.facenet import FaceNet, FaceNet64
from .ir152.ir152 import IR152
from .vgg.vgg16 import VGG16
from .vgg.vgg_face_dag import Vgg_face_dag
from .resnet.resnet50_scratch_dag import Resnet50_scratch_dag
from .inception.incv1 import InceptionResnetV1
from .get_models import get_model
# __all__ = ['ModelResult', 'FaceNet', 'FaceNet64']