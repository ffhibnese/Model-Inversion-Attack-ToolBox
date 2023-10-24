from .code.recovery import kedmi_attack
from .config import KedmiAttackConfig

def attack(args: KedmiAttackConfig):
    
    kedmi_attack(args.target_name, args.eval_name, args.ckpt_dir, args.dataset_name, args.result_dir, args.batch_size, train_gan_target_name=args.gan_target_name, device=args.device)