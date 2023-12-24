from .code.presample import presample
from .code.blackbox.blackbox_attack import mirror_blackbox_attack, MirrorBlackBoxArgs
from .code.whitebox.whitebox_attack import mirror_white_box_attack, MirrorWhiteBoxArgs
import os
from .config import MirrorBlackBoxConfig
from ...foldermanager import FolderManager
from ...models import get_model
from .code.genforce.get_genforce import get_genforce
from ...metrics import calc_knn, generate_private_feats
from .code.utils.img_utils import normalize
from torchvision import transforms
from torchvision.transforms import functional as tv_f
from ...metrics.fid.fid import calc_fid

# def blackbox_attack(genforce_name, target_name, eval_name, target_labels, work_dir, ckpt_dir, dataset_name, result_dir=None, batch_size=10, device='cpu', calc_knn=False):

def blackbox_attack(config: MirrorBlackBoxConfig):
    cache_dir = config.cache_dir
    target_name = config.target_name
    eval_name = config.eval_name
    ckpt_dir = config.ckpt_dir
    genforce_name = config.genforce_name
    batch_size = config.batch_size
    device = config.device
    target_labels = config.target_labels
    
    cache_dir = os.path.join(cache_dir, 'blackbox', f'{target_name}_{eval_name}')
    result_dir = os.path.join(config.result_dir, 'blackbox', f'{target_name}_{eval_name}')
    presample_dir = os.path.join(cache_dir, 'pre_sample', genforce_name)
    
    folder_manager = FolderManager(ckpt_dir, config.dataset_dir, cache_dir, result_dir, presample_dir=presample_dir)
    
    check_presample_dir = os.path.join(presample_dir, 'img')
    if not os.path.exists(check_presample_dir) or len(os.listdir(check_presample_dir)) == 0:
        presample(presample_dir, genforce_name, ckpt_dir, sample_num=10000, batch_size=config.batch_size, device=config.device)
    
    target_model = get_model(config.target_name, config.dataset_name, device=config.device)
    folder_manager.load_target_model_state_dict(target_model, config.dataset_name, config.target_name, device=config.device)
    
    eval_model = get_model(config.eval_name, config.dataset_name, device=config.device)
    folder_manager.load_target_model_state_dict(eval_model, config.dataset_name, config.eval_name, device=config.device)
    
    args = MirrorBlackBoxArgs(
        population=1000,
        arch_name=config.target_name,
        eval_name=eval_name,
        genforce_model_name=config.genforce_name,
        target_labels=target_labels,
        batch_size=batch_size,
        device=device,
        # calc_knn = False
    )
    
    if len(target_labels) > 0:
    
        generator, _ = get_genforce(config.genforce_name, config.device, config.ckpt_dir, use_discri=False, use_w_space=args.use_w_space, use_z_plus_space=False, repeat_w=args.repeat_w)
        
        mirror_blackbox_attack(args, generator, target_model, eval_model, folder_manager=folder_manager)
    
    print("=> Calculate the KNN Dist.")
    
    
    generate_feat_save_dir = os.path.join(config.cache_dir, config.dataset_name, config.eval_name, config.target_name)
    private_feat_save_dir = os.path.join(config.cache_dir, config.dataset_name, config.eval_name, 'private')
    
    if config.dataset_name == 'celeba':
        private_img_dir = os.path.join(config.dataset_dir, config.dataset_name, 'split', 'private', 'train')
        transform = None
    # elif config.dataset_name == 'vggface2':
    #     transform = lambda img: normalize(img * 255, config.target_name)
    else:
        raise NotImplementedError(f'dataset {config.dataset_name} is NOT supported')
    
    generate_private_feats(eval_model=eval_model, img_dir=os.path.join(result_dir, 'all_imgs'), save_dir=generate_feat_save_dir, batch_size=config.batch_size, device=config.device, transforms=None)
    generate_private_feats(eval_model=eval_model, img_dir=private_img_dir, save_dir=private_feat_save_dir, batch_size=config.batch_size, device=config.device, transforms=None, exist_ignore=True)
    
    knn_dist = calc_knn(generate_feat_save_dir, private_feat_save_dir)
    print("KNN Dist %.2f" % knn_dist)
    
def white_attack(config: MirrorBlackBoxConfig):
    # work_dir = config.cache_dir
    target_name = config.target_name
    eval_name = config.eval_name
    ckpt_dir = config.ckpt_dir
    genforce_name = config.genforce_name
    # result_dir = args.result_dir
    batch_size = config.batch_size
    dataset_name = config.dataset_name
    device = config.device
    target_labels = config.target_labels
    
    # if batch_size % len(target_labels) != 0:
    #     raise RuntimeError('batch size shoube be divisioned by number of target labels')
    
    cache_dir = os.path.join(config.cache_dir, 'whitebox', f'{target_name}_{eval_name}')
    result_dir = os.path.join(config.result_dir, 'whitebox', f'{target_name}_{eval_name}')
    presample_dir = os.path.join(cache_dir, 'pre_sample', genforce_name)
    
    folder_manager = FolderManager(ckpt_dir, config.dataset_dir, cache_dir, result_dir, presample_dir=presample_dir)
    
    check_presample_dir = os.path.join(presample_dir, 'img')
    if not os.path.exists(check_presample_dir) or len(os.listdir(check_presample_dir)) == 0:
        presample(presample_dir, genforce_name, ckpt_dir, sample_num=10000, batch_size=config.batch_size, device=config.device)
    
    target_model = get_model(config.target_name, config.dataset_name, device=config.device)
    folder_manager.load_target_model_state_dict(target_model, config.dataset_name, config.target_name, device=config.device)
    
    eval_model = get_model(config.eval_name, config.dataset_name, device=config.device)
    folder_manager.load_target_model_state_dict(eval_model, config.dataset_name, config.eval_name, device=config.device)
        
    # calc_knn = dataset_name == 'celeba'
        
    
    
    to_target_transforms = None
    
    if config.dataset_name == 'celeba':
        re_size = 64
        crop = lambda x: x[..., 20:108, 20:108]

        def trans(img):
            img = tv_f.resize(img, (128,128))
            img = crop(img)
            img = tv_f.resize(img, (re_size, re_size))
            return img
        
        to_target_transforms = trans
        
    gen_num_per_target = 5
        
    args = MirrorWhiteBoxArgs(
        arch_name=config.target_name,
        test_arch_name = config.eval_name,
        genforce_model_name=config.genforce_name,
        gen_num_per_target=gen_num_per_target,
        device=config.device,
    )
    
    batch_size = batch_size // gen_num_per_target
    
    total_acc = 0
    total_num = 0
    
    for i in range((len(target_labels)-1) // batch_size + 1):
        print(f'----------------attack batch [{i}]------------------')
        input_target_labels = target_labels[i*batch_size: min((i+1)*batch_size, len(target_labels))]
        acc = mirror_white_box_attack(args, target_model, eval_model, folder_manager = folder_manager, to_target_transforms=to_target_transforms, target_labels=input_target_labels)
        
        add_num = len(input_target_labels) * gen_num_per_target
        total_num += add_num
        total_acc += acc * add_num
        
    if total_num != 0:
        avg_acc = total_acc / total_num
        print(f'avg acc: {avg_acc: .6f}')
    
    print("=> Calculate the KNN Dist.")
    
    
    generate_feat_save_dir = os.path.join(config.cache_dir, config.dataset_name, config.eval_name, config.target_name)
    private_feat_save_dir = os.path.join(config.cache_dir, config.dataset_name, config.eval_name, 'private')
    
    if config.dataset_name == 'celeba':
        private_img_dir = os.path.join(config.dataset_dir, config.dataset_name, 'split', 'private', 'train')
        transform = None
    else:
        print(f'dataset {config.dataset_name} is NOT supported for KNN and FID')
        return
    
    generate_private_feats(eval_model=eval_model, img_dir=os.path.join(result_dir, 'all_imgs'), save_dir=generate_feat_save_dir, batch_size=config.batch_size, device=config.device, transforms=None)
    generate_private_feats(eval_model=eval_model, img_dir=private_img_dir, save_dir=private_feat_save_dir, batch_size=config.batch_size, device=config.device, transforms=None, exist_ignore=True)
    
    knn_dist = calc_knn(generate_feat_save_dir, private_feat_save_dir)
    print("KNN Dist %.2f" % knn_dist)
    
    print("=> Calculate the FID.")
    fid = calc_fid(recovery_img_path=os.path.join(result_dir, "all_imgs"),
                   private_img_path= os.path.join(config.dataset_dir, config.dataset_name, "split", "private", "train"),
                   batch_size=config.batch_size, device=config.device)
    print("FID %.2f" % fid)