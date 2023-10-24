from .code.recovery import gmi_attack
from .config import GmiAttackConfig

def attack(args: GmiAttackConfig):
    
    gmi_attack(args.target_name, args.eval_name, args.ckpt_dir, args.dataset_name, args.result_dir, args.batch_size, train_gan_target_name=args.gan_target_name, device=args.device)