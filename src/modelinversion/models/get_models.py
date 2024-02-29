from .facenet.facenet import FaceNet, FaceNet64
from .ir152.ir152 import IR152
from .vgg.vgg16 import VGG16
from .resnet.resnet50_scratch_dag import Resnet50_scratch_dag
from .inception.incv1 import InceptionResnetV1
from .vit.vit import ViT
import torchvision.models as tv_models
from .defense_wrapper import VibWrapper, TorchVisionModelWrapper
from .efficientnet.efficientnet import *

NUM_CLASSES = {
    'celeba': 1000,
    'hdceleba': 1000,
    'vggface2': 8631,
    'facescrub': 530
}


def get_model(model_name: str, dataset_name: str, device='cpu', backbone_pretrain=False, defense_type='no_defense'):
    
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()

    if dataset_name in ['celeba', 'hdceleba']:
        num_classes = 1000
    elif dataset_name == 'vggface2':
        num_classes = 8631
    elif dataset_name == 'facescrub':
        num_classes = 530
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
    # elif model_name.startswith('efficientnet'):
    #     suffix = model_name[-2:]
    #     if suffix == 'b0':
    #         model = EfficientNet_b0(num_classes, pretrained=backbone_pretrain)
    #     elif suffix == 'b1':
    #         model = EfficientNet_b1(num_classes, pretrained=backbone_pretrain)
    #     elif suffix == 'b2':
    #         model = EfficientNet_b2(num_classes, pretrained=backbone_pretrain)
    #     else:
    #         raise RuntimeError(f'model {model_name} is NOT supported')
    else:
        try:
            print('try to get model from torchvision')
            model = tv_models.__dict__[model_name](pretrained=backbone_pretrain)
            model = TorchVisionModelWrapper(model, num_classes)
        except:
            raise RuntimeError(f'model {model_name} is NOT supported')
            
    
    if defense_type.lower() in ['mid', 'vib']:
        model = VibWrapper(model, model.get_feature_dim(), num_classes)
    
    return model.to(device)