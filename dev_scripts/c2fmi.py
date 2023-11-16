import sys

sys.path.append(".")
sys.path.append("./src")
sys.path.append("./src/modelinversion")

from development_config import get_dirs
from modelinversion.attack.C2FMI.attack import attack as c2fmi_attack
from modelinversion.attack.C2FMI.config import C2FMIConfig

if __name__ == "__main__":
    dirs = get_dirs("c2fmi")
    work_dir, result_dir, ckpt_dir, dataset_dir = (
        dirs["work_dir"],
        dirs["result_dir"],
        dirs["ckpt_dir"],
        dirs["dataset_dir"],
    )

    # target name support: MobileNet
    target_name = "MobileNet"

    # eval name support: BackboneMobileFaceNet
    eval_name = "BackboneMobileFaceNet"

    # dataset name support:
    dataset_name = "FaceScrub"

    # gan dataset name support: celeba
    gan_dataset_name = "celeba"

    emb_name = "casia-InceptionResnet-Train_Acc0.984-Val_Acc0.971.pth"
    gan_name = "150000.pt"
    p2f_name = "10_pre2feat_FM2CI_keep100_loss_3.9467.pt"

    emb_backbone = "inception_resnetv1"
    tar_backbone = "mobile_net"

    batch_size = 16
    target_labels = list(range(526))
    device = "cuda:2"
    mask = 100

    config = C2FMIConfig(
        target_name=target_name,
        eval_name=eval_name,
        gan_path=gan_name,
        emb_path=emb_name,
        p2f_pth=p2f_name,
        
        ckpt_dir=ckpt_dir,
        dataset_dir=dataset_dir,
        result_dir=result_dir,
        cache_dir=work_dir,
        
        dataset_name=dataset_name,
        gan_dataset_name=gan_dataset_name,
        
        batch_size=batch_size,
        target_labels=target_labels,
        device=device,
        mask=mask,
        emb_backbone=emb_backbone,
        tar_backbone=tar_backbone,
    )

    c2fmi_attack(config)
