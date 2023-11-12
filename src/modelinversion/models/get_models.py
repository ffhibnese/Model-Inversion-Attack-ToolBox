from .facenet.facenet import FaceNet, FaceNet64
from .ir152.ir152 import IR152
from .vgg.vgg16 import VGG16
from .resnet.resnet50_scratch_dag import Resnet50_scratch_dag
from .inception.incv1 import InceptionResnetV1
import torchvision.models as tv_models

NUM_CLASSES = {
    'celeba': 1000,
    'vggface2': 8631
}


def get_model(model_name: str, dataset_name: str, device='cpu', backbone_pretrain=False):
    
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()

    if dataset_name == 'celeba':
        num_classes = 1000
    elif dataset_name == 'vggface2':
        num_classes = 8631
    elif dataset_name == 'imagenet':
        return tv_models.__dict__[model_name](pretrained=True)
    else:
        raise RuntimeError(f'dataset {dataset_name} is not supported')
    
    
    if model_name == 'vgg16':
        model = VGG16(num_classes, pretrained=backbone_pretrain)
    elif model_name == 'ir152':
        model = IR152(num_classes)
    elif model_name == 'facenet':
        model = FaceNet(num_classes)
    elif model_name == 'facenet64':
        model = FaceNet64(num_classes)
    elif model_name == 'resnet50_scratch_dag':
        model = Resnet50_scratch_dag(num_classes)
    elif model_name == 'inception_resnetv1':
        model = InceptionResnetV1(num_classes)
    else:
        raise RuntimeError(f'model {model_name} is NOT supported')
    
    return model.to(device)