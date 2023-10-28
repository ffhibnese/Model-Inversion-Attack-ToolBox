from .facenet.facenet import FaceNet, FaceNet64
from .ir152.ir152 import IR152
from .vgg.vgg16 import VGG16



def get_model(model_name, dataset_name, device='cpu'):

    if dataset_name == 'celeba':
        num_classes = 1000
    else:
        raise RuntimeError(f'dataset {dataset_name} is not supported')
    
    
    if model_name == 'vgg16':
        model = VGG16(num_classes)
    elif model_name == 'ir152':
        model = IR152(num_classes)
    elif model_name == 'facenet':
        model = FaceNet(num_classes)
    elif model_name == 'facenet64':
        model = FaceNet64(num_classes)
    
    else:
        raise RuntimeError(f'model {model_name} is NOT supported')
    
    return model.to(device)