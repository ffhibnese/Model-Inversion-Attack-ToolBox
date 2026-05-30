"""
Unified path configuration for all attacks in ffhq256_facescrub224.

Usage:
    from attack_paths import get_attack_paths

    # For baseline (no defense)
    paths = get_attack_paths('gmi', 'no')

    # For defended model
    paths = get_attack_paths('gmi', 'vib0.01')
"""

from dataclasses import dataclass, field
from typing import Optional, List

# =============================================================================
# Constants - Common paths shared across all attacks
# =============================================================================

_USR = '<usrname>'

# Root directories
RESP_ROOT = f'/mnt/data/{_USR}/mywork/lora_defense'
EXP_SETTING_ROOT = f'{RESP_ROOT}/test_lora/ffhq256_facescrub224'
DATASET_ROOT = f'/mnt/data/{_USR}/datasets'

_TARGET_MODEL_PATH = f'{EXP_SETTING_ROOT}/result_classifier/train_facescrub224_resnet152{{tag}}/facescrub224_resnet152{{tag}}.pth'
_EVAL_MODEL_PATH = f'{EXP_SETTING_ROOT}/result_classifier/train_facescrub224_maxvit_t/facescrub224_maxvit_t.pth'
_EVAL_DATASET_PATH = f'{DATASET_ROOT}/facescrub/'

_STYLEGAN2ADA_PATH = f'{RESP_ROOT}/test_resp/stylegan2-ada-pytorch'
_STYLEGAN2ADA_CKPT_PATH = f'{RESP_ROOT}/checkpoints_v2/stylegan2ada/ffhq.pkl'

_GMI_GAN_PATH = f'{RESP_ROOT}/checkpoints_v2/gmi/gmi_ffhq256'

_DATASET_PATH = f'{DATASET_ROOT}/ffhq256'


# =============================================================================
# Dataclass
# =============================================================================

@dataclass
class AttackPaths:
    """Path configuration for a model inversion attack.

    Attributes:
        name: Attack name (e.g., 'gmi', 'c2f', 'lokt')
        attack_file: Primary attack script filename
        tag: Raw tag input (e.g., 'no', 'vib0.01')

        # Always present
        target_model_ckpt_path: Path to target classifier checkpoint
        eval_model_ckpt_path: Path to evaluation model checkpoint
        eval_dataset_path: Path to evaluation dataset
        experiment_dir: Output directory for attack results

        # GAN-based attacks
        generator_ckpt_path: Path to generator checkpoint (if applicable)
        discriminator_ckpt_path: Path to discriminator checkpoint (if applicable)

        # StyleGAN-based attacks
        stylegan2ada_path: Path to StyleGAN2-ADA repository
        stylegan2ada_ckpt_path: Path to StyleGAN2-ADA checkpoint

        # GAN training artifacts
        gan_experiment_dir: Output directory for GAN training (if applicable)

        # LOKT-specific
        lokt_aug_model_ckpt_paths: Paths to augmentation model checkpoints
        lokt_dataset_path: Path to LOKT training dataset
        lokt_train_dataset_path: Path to generated training dataset

        # C2F-specific
        c2f_embed_model_ckpt_path: Path to embedding model
        c2f_pred_mapping_ckpt_path: Path to prediction mapping model

        # LOMMA-specific
        lomma_aug_model_ckpt_paths: Paths to augmentation model checkpoints
        lomma_public_dataset_path: Path to public dataset for feature statistics

        # PPDG-specific
        ppdg_tuning_feature_extractor_path: Path to VGG16 feature extractor

        # Prerequisites (scripts that must run before attack, in order)
        prerequisites: list[str] = field(default_factory=list)
    """

    # Identity
    name: str
    tag: str
    attack_file: str = ''  # Set in if/elif branches

    # Always present
    target_model_ckpt_path: str = ''
    eval_model_ckpt_path: str = ''
    eval_dataset_path: str = ''
    experiment_dir: str = ''

    # GAN-based attacks
    generator_ckpt_path: Optional[str] = None
    discriminator_ckpt_path: Optional[str] = None

    # StyleGAN-based attacks
    stylegan2ada_path: Optional[str] = None
    stylegan2ada_ckpt_path: Optional[str] = None

    # GAN training artifacts
    gan_experiment_dir: Optional[str] = None

    # LOKT-specific
    lokt_aug_model_ckpt_paths: Optional[List[str]] = None
    lokt_dataset_path: Optional[str] = None
    lokt_train_dataset_path: Optional[str] = None

    # C2F-specific
    c2f_embed_model_ckpt_path: Optional[str] = None
    c2f_pred_mapping_ckpt_path: Optional[str] = None

    # LOMMA-specific
    lomma_aug_model_ckpt_paths: Optional[List[str]] = None
    lomma_public_dataset_path: Optional[str] = None

    # PPDG-specific
    ppdg_tuning_feature_extractor_path: Optional[str] = None

    # Prerequisites
    prerequisites: List[str] = field(default_factory=list)

    # For GAN training scripts (ked/gan.py, plg/gan.py, lokt/gan.py)
    gan_batch_size: Optional[int] = None
    gan_max_iters: Optional[int] = None

    # For cls.py (c2f)
    cls_batch_size: Optional[int] = None
    cls_embed_model_ckpt_path: Optional[str] = None

    # For lokt/ds.py
    ds_batch_size: Optional[int] = None

    # For lokt/cls.py
    cls_train_batch_size: Optional[int] = None

    # For lomma/surr/eff0.py
    distill_model_names: Optional[List[str]] = None
    distill_batch_size: Optional[int] = None
    distill_epoch_num: Optional[int] = None

    # CUDA device (default '0', override via CUDA_VISIBLE_DEVICES env var)
    cuda_device: str = '0'


# =============================================================================
# Helpers
# =============================================================================

def _format_tag(tag: str) -> str:
    """Convert tag to path suffix: 'no' -> '', others -> '_tag'"""
    return '' if tag == 'no' else f'_{tag}'


def _build_lomma_aug_model_paths(tag: str) -> List[str]:
    """Build LOMMA augmentation model paths."""
    return [
        f'./results_attack/distill_efficientnet_b0_ir152_{tag}/ffhq256_efficientnet_b0_facescrub224_{tag}.pth',
        f'./results_attack/distill_efficientnet_b1_ir152_{tag}/ffhq256_efficientnet_b1_facescrub224_{tag}.pth',
        f'./results_attack/distill_efficientnet_b2_ir152_{tag}/ffhq256_efficientnet_b2_facescrub224_{tag}.pth',
    ]


# =============================================================================
# Factory
# =============================================================================

def get_attack_paths(name: str, tag: str) -> AttackPaths:
    """Unified factory for all attack paths.

    Args:
        name: Attack name ('c2f', 'gmi', 'if', 'ked', 'lokt', 'lomma_lgmi',
              'lomma_lked', 'mb', 'mirror', 'plg', 'ppa', 'ppdg', 'brep')
        tag: Defense tag ('no' for baseline, or 'vib0.01', 'tl0.5', etc.)

    Returns:
        AttackPaths dataclass with all path configurations

    Raises:
        ValueError: If attack name is unknown
    """
    tag_str = _format_tag(tag)

    # Base paths always present
    base = AttackPaths(
        name=name,
        tag=tag,
        target_model_ckpt_path=_TARGET_MODEL_PATH.format(tag=tag_str),
        eval_model_ckpt_path=_EVAL_MODEL_PATH,
        eval_dataset_path=_EVAL_DATASET_PATH,
    )

    # -------------------------------------------------------------------------
    # C2F: requires cls.py first, then c2f.py
    # -------------------------------------------------------------------------
    if name == 'c2f':
        base.attack_file = 'c2f.py'
        base.prerequisites = ['cls.py']
        base.experiment_dir = f'./attack/c2f_ir152{tag_str}'
        base.stylegan2ada_path = _STYLEGAN2ADA_PATH
        base.stylegan2ada_ckpt_path = _STYLEGAN2ADA_CKPT_PATH
        base.c2f_embed_model_ckpt_path = (
            f'/mnt/data/{_USR}/mywork/lora_defense/checkpoints_v2/c2f/casia_incv1.pth'
        )
        base.c2f_pred_mapping_ckpt_path = (
            f'./results_mapping/c2f/ffhq256_facescrub224/ir152{tag_str}/mapping.pth'
        )
        base.cls_batch_size = 128
        base.cls_embed_model_ckpt_path = base.c2f_embed_model_ckpt_path
        return base

    # -------------------------------------------------------------------------
    # GMI: direct attack, no prerequisites
    # -------------------------------------------------------------------------
    elif name == 'gmi':
        base.attack_file = 'att.py'
        base.prerequisites = []
        base.experiment_dir = f'./results_attack/gmi_ir152_{tag}'
        base.generator_ckpt_path = f'{_GMI_GAN_PATH}_G.pth'
        base.discriminator_ckpt_path = f'{_GMI_GAN_PATH}_D.pth'
        return base

    # -------------------------------------------------------------------------
    # KED: requires gan.py first, then att.py
    # -------------------------------------------------------------------------
    elif name == 'ked':
        base.attack_file = 'att.py'
        base.prerequisites = ['gan.py']
        base.experiment_dir = f'./attack_result/kedmi_ir152_{tag}_100'
        base.generator_ckpt_path = (
            f'./results_gan/kedmi_ffhq64_facescrub64_ir152{tag_str}_gan/G.pth'
        )
        base.discriminator_ckpt_path = (
            f'./results_gan/kedmi_ffhq64_facescrub64_ir152{tag_str}_gan/D.pth'
        )
        base.gan_experiment_dir = (
            f'./results_gan/kedmi_ffhq64_facescrub64_ir152{tag_str}_gan'
        )
        base.gan_batch_size = 32  # halved from 64
        base.gan_max_iters = 50000
        return base

    # -------------------------------------------------------------------------
    # PLG: requires gan.py first, then att.py
    # -------------------------------------------------------------------------
    elif name == 'plg':
        base.attack_file = 'att.py'
        base.prerequisites = ['gan.py']
        base.experiment_dir = f'./results_attack/plgmi_resnet152{tag_str}'
        base.generator_ckpt_path = (
            f'./results_gan/plg_ffhq256_facescrub256_resnet152{tag_str}_gan/G.pth'
        )
        base.gan_experiment_dir = (
            f'./results_gan/plg_ffhq256_facescrub256_resnet152{tag_str}_gan'
        )
        base.gan_batch_size = 16  # halved from 32
        base.gan_max_iters = 100000
        return base

    # -------------------------------------------------------------------------
    # LOKT: requires gan.py -> ds.py -> cls.py first, then att.py
    # -------------------------------------------------------------------------
    elif name == 'lokt':
        base.attack_file = 'att.py'
        base.prerequisites = ['gan.py', 'ds.py', 'cls.py']
        base.experiment_dir = f'./results_attack/lokt_{tag_str}'
        base.generator_ckpt_path = (
            f'./gan/lokt_ffhq256_facescrub224_ir152{tag_str}_gan/G.pth'
        )
        base.gan_experiment_dir = (
            f'./gan/lokt_ffhq256_facescrub224_ir152{tag_str}_gan'
        )
        base.lokt_aug_model_ckpt_paths = [
            f'./classifier/lokt_ffhq256_facescrub224_ir152{tag_str}/densenet121/facescrub224_densenet121.pth',
            f'./classifier/lokt_ffhq256_facescrub224_ir152{tag_str}/densenet161/facescrub224_densenet161.pth',
            f'./classifier/lokt_ffhq256_facescrub224_ir152{tag_str}/densenet169/facescrub224_densenet169.pth',
        ]
        base.lokt_dataset_path = _DATASET_PATH
        base.lokt_train_dataset_path = (
            f'./dataset/lokt_ffhq256_facescrub224_ir152{tag_str}_dataset/dataset.pt'
        )
        base.ds_batch_size = 100  # halved from 200
        base.cls_train_batch_size = 64  # halved from 128
        base.gan_batch_size = 32  # halved from 64
        base.gan_max_iters = 105000
        return base

    # -------------------------------------------------------------------------
    # LOMMA/LGMI: requires surr/eff0.py first, uses gmi's GAN
    # -------------------------------------------------------------------------
    elif name == 'lomma_lgmi':
        base.attack_file = 'lgmi.py'
        base.prerequisites = ['surr/eff0.py']
        base.experiment_dir = f'./results_attack/lommagmi_ir152_{tag}'
        # Reuses gmi GAN
        base.generator_ckpt_path = f'{_GMI_GAN_PATH}_G.pth'
        base.discriminator_ckpt_path = f'{_GMI_GAN_PATH}_D.pth'
        base.lomma_aug_model_ckpt_paths = _build_lomma_aug_model_paths(tag)
        base.lomma_public_dataset_path = _DATASET_PATH
        base.distill_model_names = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2']
        base.distill_batch_size = 64  # halved from 128
        base.distill_epoch_num = 100
        return base

    # -------------------------------------------------------------------------
    # LOMMA/LKED: requires surr/eff0.py first, uses ked's GAN
    # -------------------------------------------------------------------------
    elif name == 'lomma_lked':
        base.attack_file = 'lked.py'
        base.prerequisites = ['surr/eff0.py']
        base.experiment_dir = f'./results_attack/lommaked_ir152_{tag}'
        # Reuses ked GAN (from ../ked/results_gan/)
        base.generator_ckpt_path = (
            f'../ked/results_gan/kedmi_ffhq64_facescrub64_ir152{tag_str}_gan/G.pth'
        )
        base.discriminator_ckpt_path = (
            f'../ked/results_gan/kedmi_ffhq64_facescrub64_ir152{tag_str}_gan/D.pth'
        )
        base.lomma_aug_model_ckpt_paths = _build_lomma_aug_model_paths(tag)
        base.lomma_public_dataset_path = _DATASET_PATH
        base.distill_model_names = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2']
        base.distill_batch_size = 64  # halved from 128
        base.distill_epoch_num = 100
        return base

    # -------------------------------------------------------------------------
    # IF: direct StyleGAN attack, no prerequisites
    # -------------------------------------------------------------------------
    elif name == 'if':
        base.attack_file = 'att.py'
        base.prerequisites = []
        base.experiment_dir = f'./attack_result/if_resnet152{tag_str}'
        base.stylegan2ada_path = _STYLEGAN2ADA_PATH
        base.stylegan2ada_ckpt_path = _STYLEGAN2ADA_CKPT_PATH
        return base

    # -------------------------------------------------------------------------
    # PPA: direct StyleGAN attack, no prerequisites
    # -------------------------------------------------------------------------
    elif name == 'ppa':
        base.attack_file = 'att.py'
        base.prerequisites = []
        base.experiment_dir = f'./attack_result/ppa_resnet152{tag_str}'
        base.stylegan2ada_path = _STYLEGAN2ADA_PATH
        base.stylegan2ada_ckpt_path = _STYLEGAN2ADA_CKPT_PATH
        return base

    # -------------------------------------------------------------------------
    # PPDG: direct StyleGAN attack, no prerequisites
    # -------------------------------------------------------------------------
    elif name == 'ppdg':
        base.attack_file = 'att.py'
        base.prerequisites = []
        base.experiment_dir = f'./attack_result/ppdg_resnet152{tag_str}'
        base.stylegan2ada_path = _STYLEGAN2ADA_PATH
        base.stylegan2ada_ckpt_path = _STYLEGAN2ADA_CKPT_PATH
        base.ppdg_tuning_feature_extractor_path = (
            f'/mnt/data/{_USR}/mywork/lora_defense/checkpoints_v2/ppdg/vgg16.pt'
        )
        return base

    # -------------------------------------------------------------------------
    # MIRROR: direct StyleGAN attack, no prerequisites
    # -------------------------------------------------------------------------
    elif name == 'mirror':
        base.attack_file = 'att.py'
        base.prerequisites = []
        base.experiment_dir = f'./attack_result/if_resnet152{tag_str}'  # Note: same as if
        base.stylegan2ada_path = _STYLEGAN2ADA_PATH
        base.stylegan2ada_ckpt_path = _STYLEGAN2ADA_CKPT_PATH
        return base

    # -------------------------------------------------------------------------
    # BREP: direct attack, no prerequisites, uses gmi's GAN
    # -------------------------------------------------------------------------
    elif name == 'brep':
        base.attack_file = 'brepmi.py'
        base.prerequisites = []
        base.experiment_dir = f'./results_attack/brep_ir152{tag_str}'
        # Uses gmi GAN
        base.generator_ckpt_path = f'{_GMI_GAN_PATH}_G.pth'
        base.discriminator_ckpt_path = f'{_GMI_GAN_PATH}_D.pth'
        return base

    # -------------------------------------------------------------------------
    # MB (Mirror Black): direct StyleGAN attack, no prerequisites
    # -------------------------------------------------------------------------
    elif name == 'mb':
        base.attack_file = 'mirror_black.py'
        base.prerequisites = []
        base.experiment_dir = f'./attack_result/mb_resnet152{tag_str}'
        base.stylegan2ada_path = _STYLEGAN2ADA_PATH
        base.stylegan2ada_ckpt_path = _STYLEGAN2ADA_CKPT_PATH
        return base

    # -------------------------------------------------------------------------
    # Unknown
    # -------------------------------------------------------------------------
    else:
        known_attacks = [
            'c2f', 'gmi', 'if', 'ked', 'lokt',
            'lomma_lgmi', 'lomma_lked',
            'mb', 'mirror', 'plg', 'ppa', 'ppdg', 'brep'
        ]
        raise ValueError(
            f"Unknown attack: {name!r}. Expected one of {known_attacks}"
        )


# =============================================================================
# Convenience exports
# =============================================================================

ALL_ATTACKS = [
    'c2f', 'gmi', 'if', 'ked', 'lokt',
    'lomma_lgmi', 'lomma_lked',
    'mb', 'mirror', 'plg', 'ppa', 'ppdg', 'brep'
]

ALL_TAGS = [
    'no', 'vib0.01', 'bido0.01_0.1_pretrain',
    'ls0.05', 'tl0.5', 'rolss0.0_2'
]