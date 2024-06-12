from dataclasses import dataclass,field
from abc import ABC,abstractmethod
from models import *
from sampler import *
from attack import *
from datasets import *
from torch.utils.data import Dataset

@dataclass
class BaseAttackConfig(ABC):
    experiment_dir: str
    generator_ckpt_path: str
    discriminator_ckpt_path: str
    target_model_ckpt_path: str
    eval_model_ckpt_path: str
    eval_dataset_path: str
    
    batch_size: int
    device: torch.device
    gpu_devices: list[int] = field(default_factory=list)
    attack_targets: list[int] = field(default_factory=list)
    
    def _parse_models(self):
        target_model = auto_classifier_from_pretrained(self.target_model_ckpt_path)
        eval_model = auto_classifier_from_pretrained(
            self.eval_model_ckpt_path, register_last_feature_hook=True
        )
        generator = auto_generator_from_pretrained(self.generator_ckpt_path)

        target_model = nn.DataParallel(target_model, device_ids=self.gpu_devices).to(self.device)
        eval_model = nn.DataParallel(eval_model, device_ids=self.gpu_devices).to(self.device)
        generator = nn.DataParallel(generator, device_ids=self.gpu_devices).to(self.device)
        target_model.eval()
        eval_model.eval()
        generator.eval()
        return target_model,eval_model,generator
    
    @abstractmethod
    def default_params(self):
        pass

    @abstractmethod
    def get_attacker(self):
        pass

class GmiAttackConfig(BaseAttackConfig):
    z_dim: int = 100
    optimize_num: int = 50
    
    def default_params(self):
        # prepare models
        
        self.latents_sampler = SimpleLatentsSampler(self.z_dim, self.batch_size)
        self.target_model,self.eval_model,self.generator = self._parse_models()
        discriminator = auto_discriminator_from_pretrained(self.discriminator_ckpt_path)
        discriminator = nn.DataParallel(discriminator, device_ids=self.gpu_devices).to(self.device)
        discriminator.eval()
        self.discriminator = discriminator
        
        # prepare eval dataset

        self.eval_dataset = CelebA112(
            self.eval_dataset_path,
            output_transform=ToTensor(),
        )
        
        # prepare optimization

        optimization_config = SimpleWhiteBoxOptimizationConfig(
            experiment_dir=self.experiment_dir,
            device=self.device,
            optimizer='SGD',
            optimizer_kwargs={'lr': 0.02, 'momentum': 0.9},
            iter_times=1500,
        )

        identity_loss_fn = ImageAugmentClassificationLoss(
            classifier=self.target_model, loss_fn='ce', create_aug_images_fn=None
        )

        discriminator_loss_fn = GmiDiscriminatorLoss(discriminator)

        loss_fn = ComposeImageLoss(
            [identity_loss_fn, discriminator_loss_fn], weights=[100, 1]
        )

        self.optimization_fn = SimpleWhiteBoxOptimization(
            optimization_config, generator, loss_fn
        )
    
    def get_attacker(self,
                     save_optimized_images: bool=True, 
                     save_final_images: bool=False,
                     eval_metrics: list=[],
                     eval_optimized_result:bool=True,
                     eval_final_result:bool=False):
        
        # prepare attack

        attack_config = ImageClassifierAttackConfig(
            # attack args
            self.latents_sampler,
            optimize_num=self.optimize_num,
            optimize_batch_size=self.batch_size,
            optimize_fn=self.optimization_fn,
            
            # save path args
            save_dir=self.experiment_dir,
            save_optimized_images=save_optimized_images,
            save_final_images=save_final_images,
            
            # metric args
            eval_metrics=eval_metrics,
            eval_optimized_result=eval_optimized_result,
            eval_final_result=eval_final_result,
        )

        return ImageClassifierAttacker(attack_config)