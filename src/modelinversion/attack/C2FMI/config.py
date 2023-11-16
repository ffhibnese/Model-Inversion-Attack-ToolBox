from dataclasses import dataclass, field

@dataclass
class C2FMIConfig:
    
    target_name: str
    eval_name: str
    gan_path: str
    emb_path: str   # path of the embedding model
    p2f_pth: str    # path of the inverse model
    
    ckpt_dir: str
    dataset_dir: str
    result_dir: str
    cache_dir: str
    
    dataset_name: str
    gan_dataset_name: str
    
    batch_size: int = 16
    target_labels: list = field(default_factory = lambda : list(range(300)))
    device: str = 'cpu'
    
    n_mean_latent: int = 10000
    img_size: int = 128
    mask: int = 100
    tar_classes: int = 526          # total target label's num
    emb_classes: int = 10575        # 提取出来的特征数
    init_lr: float = 0.02
    step: int = 50                  # 第一阶段迭代次数
    face_shape: list = field(default_factory = lambda : [160, 160])   # 图片分辨率
    
    emb_backbone: str = 'inception_resnetv1'    #backbone of embedding model
    tar_backbone: str = 'mobile_net'            #backbone of target model
    eva_backbone: str = 'mobile_facenet'
    only_best: bool = True
    
    