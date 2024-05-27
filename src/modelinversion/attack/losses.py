from abc import abstractmethod, ABC
from collections import OrderedDict
from typing import Callable, Any, Optional, Iterable, Sequence


import torch
from torch import nn, Tensor, LongTensor
from torch.nn import functional as F
from ..utils import (
    TorchLoss,
    reparameterize,
    DeepInversionBNFeatureHook,
    traverse_module,
)
from ..models import (
    BaseImageClassifier,
    GmiDiscriminator64,
    GmiDiscriminator256,
    KedmiDiscriminator64,
    KedmiDiscriminator256,
    HOOK_NAME_FEATURE,
)

from ..sampler import LayeredFlowMiner, MixtureOfGMM


class BaseImageLoss(ABC):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def __call__(self, images: Tensor, labels: LongTensor, *args, **kwargs):
        pass

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return self.__class__.__name__


class ImageAugmentClassificationLoss(BaseImageLoss):

    def __init__(
        self,
        classifier: BaseImageClassifier,
        loss_fn: str | Callable[[Tensor, LongTensor], Tensor] = 'cross_entropy',
        create_aug_images_fn: Optional[Callable[[Tensor], Iterable[Tensor]]] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if create_aug_images_fn is None:
            create_aug_images_fn = lambda x: [x]

        self.create_aug_images_fn = create_aug_images_fn
        self.classifier = classifier
        self.loss_fn = TorchLoss(loss_fn, *args, **kwargs)

    def __call__(self, images, labels, *args, **kwargs):
        acc = 0
        loss = 0
        total_num = 0
        for aug_images in self.create_aug_images_fn(images):
            total_num += 1
            conf, _ = self.classifier(aug_images)
            pred_labels = torch.argmax(conf, dim=-1)
            loss += self.loss_fn(conf, labels)
            # print(pred_labels)
            # print(labels)
            # exit()
            acc += (pred_labels == labels).float().mean().item()

        return loss, OrderedDict(
            [['classification loss', loss.item()], ['target acc', acc / total_num]]
        )

    def __repr__(self) -> str:
        return f'classifier loss'


class ClassificationWithFeatureDistributionLoss(ImageAugmentClassificationLoss):

    def __init__(
        self,
        classifier: BaseImageClassifier,
        feature_mean: Tensor,
        feature_std: Tensor,
        classification_loss_fn: (
            str | Callable[[Tensor, LongTensor], Tensor]
        ) = 'cross_entropy',
        create_aug_images_fn: Optional[Callable[[Tensor], Iterable[Tensor]]] = None,
        feature_loss_weight: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            classifier, classification_loss_fn, create_aug_images_fn, *args, **kwargs
        )

        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.feature_loss_weight = feature_loss_weight

    def _sample_distribution(self):
        return reparameterize(self.feature_mean, self.feature_std)

    def __call__(self, images, labels, *args, **kwargs):

        acc = 0
        iden_loss = 0
        feature_loss = 0
        total_num = 0
        bs = len(images)
        for aug_images in self.create_aug_images_fn(images):
            total_num += 1
            conf, info_dict = self.classifier(aug_images)
            if HOOK_NAME_FEATURE not in info_dict:
                raise RuntimeError(
                    f'The addition info that the model outputs do not contains {HOOK_NAME_FEATURE}'
                )
            pred_labels = torch.argmax(conf, dim=-1)
            iden_loss += self.loss_fn(conf, labels)

            feature_dist_samples = self._sample_distribution()
            feature_loss += torch.mean(
                (
                    info_dict[HOOK_NAME_FEATURE].view(bs, -1)
                    - feature_dist_samples.view(1, -1)
                ).pow(2)
            )
            acc += (pred_labels == labels).float().mean().item()

        loss = iden_loss + self.feature_loss_weight * feature_loss

        return loss, OrderedDict(
            [
                ['loss', loss.item()],
                ['classification loss', iden_loss.item()],
                ['feature loss', feature_loss.item()],
                ['target acc', acc / total_num],
            ]
        )

    def __repr__(self) -> str:
        return 'classifier loss'


class GmiDiscriminatorLoss(BaseImageLoss):

    def __init__(
        self, discriminator: GmiDiscriminator64 | GmiDiscriminator256, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.discriminator = discriminator

    def __call__(self, images, labels, *args, **kwargs):
        dis_res = self.discriminator(images)
        loss = -dis_res.mean()
        return loss, {'discriminator loss': loss.item()}

    def __repr__(self) -> str:
        return 'gmi discriminator loss'


class KedmiDiscriminatorLoss(BaseImageLoss):

    def __init__(
        self,
        discriminator: KedmiDiscriminator64 | KedmiDiscriminator256,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.discriminator = discriminator

    def __call__(self, images, labels, *args, **kwargs):
        _, dis_res = self.discriminator(images)
        logsumup = torch.logsumexp(dis_res, dim=-1)
        # loss = - dis_res.mean()
        loss = torch.mean(F.softplus(logsumup)) - torch.mean(logsumup)
        return loss, {'discriminator loss': loss.item()}

    def __repr__(self) -> str:
        return 'kedmi discriminator loss'


class VmiLoss(BaseImageLoss):

    def __init__(
        self,
        classifier: BaseImageClassifier,
        miner: nn.Module,
        batch_size: int,
        device: torch.device,
        weights: dict, 
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lambda_attack = weights['lambda_attack']
        self.lambda_miner_entropy = weights['lambda_miner_entropy']
        self.lambda_kl = weights['lambda_kl']
        self.classifier = classifier
        self.miner = miner
        self.batch_size = (batch_size,)
        self.device = device

    def extract_feat(self, x, mb=100):
        assert x.min() >= 0  # in case passing in x in [-1,1] by accident
        zs = []
        C, H, W = x.shape[1:]
        print(C, H, W)
        for start in range(0, len(x), mb):
            _x = x[start : start + mb]
            zs.append(self.classifier((_x.view(_x.size(0), C, H, W) - 0.5) / 0.5))
        return torch.cat(zs)

    def attack_criterion(self, lsm, target):
        true_dist = torch.zeros_like(lsm)
        true_dist.fill_(0)
        true_dist.scatter_(1, target.data.unsqueeze(1), 1.0)
        return torch.mean(torch.sum(-true_dist * lsm, dim=-1))

    def gaussian_logp(self, mean, logstd, x, detach=False):
        """
        lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
                k = 1 (Independent)
                Var = logstd ** 2
        """
        import numpy as np

        c = np.log(2 * np.pi)
        v = -0.5 * (logstd * 2.0 + ((x - mean) ** 2) / torch.exp(logstd * 2.0) + c)
        if detach:
            v = v.detach()
        return v

    def __call__(self, images: Tensor, labels: LongTensor, *args, **kwargs):
        lsm = self.extract_feat(images / 2 + 0.5)
        loss_attack = 0
        return_dict = OrderedDict()
        if self.lambda_attack > 0:
            loss_attack = self.attack_criterion(lsm, labels)
            return_dict['attack loss'] = loss_attack.item()

        loss_miner_entropy = 0
        if self.lambda_miner_entropy > 0:
            loss_miner_entropy = -self.miner.entropy()
            return_dict['miner_entropy loss'] = loss_miner_entropy.item()

        loss_kl = 0
        if self.lambda_kl > 0:
            if isinstance(self.miner, MixtureOfGMM):
                for gmm in self.miner.gmms:
                    samples = gmm(
                        torch.randn(self.batchSize, gmm.nz0).to(self.device).double()
                    )
                    loss_kl += torch.mean(
                        gmm.logp(samples)
                        - self.gaussian_logp(
                            torch.zeros_like(samples),
                            torch.zeros_like(samples),
                            samples,
                        ).sum(-1)
                    )
                loss_kl /= len(self.miner.gmms)
            elif isinstance(self.miner, LayeredFlowMiner):
                # 1/L * \sum_l KL(Flow_l || N(0,1))
                for flow in self.miner.flow_miners:
                    samples = flow(
                        torch.randn(self.batchSize, flow.nz0).to(self.device).double()
                    )
                    loss_kl += torch.mean(
                        flow.logp(samples)
                        - self.gaussian_logp(
                            torch.zeros_like(samples),
                            torch.zeros_like(samples),
                            samples,
                        ).sum(-1)
                    )
                loss_kl /= len(self.miner.flow_miners)
            return_dict['kl loss'] = loss_kl.item()

        loss = (self.lambda_attack * loss_attack
                    + self.lambda_miner_entropy * loss_miner_entropy
                    + self.lambda_kl * loss_kl)
        return_dict['loss'] = loss.item()
        return loss, return_dict

    def __repr__(self) -> str:
        return 'vmi loss'


class ComposeImageLoss(BaseImageLoss):

    def __init__(
        self,
        losses: list[BaseImageLoss],
        weights: Optional[list[float]] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if weights is None:
            weights = [1] * len(losses)

        if len(losses) == 0:
            raise RuntimeError(f'losses should be at least one function.')

        if len(weights) != len(losses):
            raise RuntimeError(
                f'Expect the equal length of losses and weights, but found the fronter {len(losses)} and the latter {len(weights)}.'
            )

        self.losses = losses
        self.weights = weights

    def __call__(self, images: Tensor, labels: LongTensor, *args, **kwargs):

        compose_loss = 0.0
        return_dict = OrderedDict()
        return_dict['compose loss'] = 0

        for i, (lossfn, weight) in enumerate(zip(self.losses, self.weights)):

            loss = lossfn(images, labels, *args, **kwargs)
            if not isinstance(loss, Tensor):
                loss, single_dict = loss
                # for k, v in single_dict.items():
                #     return_dict[k] = v
                if single_dict is not None:
                    k = f'{i} - {lossfn}'
                    return_dict[k] = single_dict
            compose_loss += weight * loss

        return_dict['compose loss'] = compose_loss.item()

        return compose_loss, return_dict

    def __repr__(self) -> str:
        return 'compose loss'


class ImagePixelPriorLoss(BaseImageLoss):

    def __init__(
        self, l1_weight: float = 0, l2_weight: float = 0, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def __call__(self, images: Tensor, labels: LongTensor, *args, **kwargs):
        l1_loss = images.abs().mean()
        l2_loss = torch.norm(images).mean()
        loss = l1_loss * self.l1_weight + l2_loss * self.l2_weight
        return loss, OrderedDict(
            [
                ['l1 loss', l1_loss.item()],
                ['l2 loss', l2_loss.item()],
                ['loss', loss.item()],
            ]
        )

    def __repr__(self) -> str:
        return 'image pixel loss'


class ImageVariationPriorLoss(BaseImageLoss):

    def __init__(
        self, l1_weight: float = 0, l2_weight: float = 0, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def __call__(self, images: Tensor, labels: LongTensor, *args, **kwargs):

        diff1 = images[..., :, :-1] - images[..., :, 1:]
        diff2 = images[..., :-1, :] - images[..., 1:, :]
        diff3 = images[..., 1:, :-1] - images[..., :-1, 1:]
        diff4 = images[..., :-1, :-1] - images[..., 1:, 1:]

        loss_var_l2 = (
            torch.norm(diff1)
            + torch.norm(diff2)
            + torch.norm(diff3)
            + torch.norm(diff4)
        ) / 4
        loss_var_l1 = (
            diff1.abs().mean()
            + diff2.abs().mean()
            + diff3.abs().mean()
            + diff4.abs().mean()
        ) / 4
        loss = loss_var_l1 * self.l1_weight + loss_var_l2 * self.l2_weight
        return loss, OrderedDict(
            [
                ['l1 var loss', loss_var_l1.item()],
                ['l2 var loss', loss_var_l2.item()],
                ['loss', loss.item()],
            ]
        )

    def __repr__(self) -> str:
        return 'image variation loss'


class DeepInversionBatchNormPriorLoss(BaseImageLoss):

    def __init__(self, model, first_bn_weight=10, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.feature_hooks: list[DeepInversionBNFeatureHook] = []

        self.first_bn_weight = first_bn_weight

        def _find_bn_fn(module):
            if isinstance(module, nn.BatchNorm2d):
                self.feature_hooks.append(DeepInversionBNFeatureHook(module))

        traverse_module(model, _find_bn_fn, call_middle=True)

        if len(self.feature_hooks) == 0:
            raise RuntimeError(f'The model do not have BN layers.')

    def __call__(self, images: Tensor, labels: LongTensor, *args, **kwargs):

        r_features_losses = [hook.get_feature() for hook in self.feature_hooks]
        r_features_losses = [l.sum() for l in r_features_losses if l is not None]
        r_features_losses[0] *= self.first_bn_weight

        loss = sum(r_features_losses)
        return loss, OrderedDict(loss=loss.item())

    def __repr__(self) -> str:
        return 'deep inversion BN loss'


class MultiModelOutputKLLoss(BaseImageLoss):

    def __init__(
        self,
        teacher: BaseImageClassifier,
        students: BaseImageClassifier | list[BaseImageClassifier],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.teacher = teacher
        if isinstance(students, nn.Module):
            students = [students]
        self.students = students

    def __call__(self, images: Tensor, labels: LongTensor, *args, **kwargs):
        T = 3.0
        output_teacher = self.teacher(images)
        if not isinstance(output_teacher, Tensor):
            output_teacher = output_teacher[0]
        Q = nn.functional.softmax(output_teacher / T, dim=1)
        Q = torch.clamp(Q, 0.01, 0.99)

        loss = 0
        for student in self.students:
            output_student = student(images)
            if not isinstance(output_student, Tensor):
                output_student = output_student[0]

            # Jensen Shanon divergence:
            # another way to force KL between negative probabilities
            P = nn.functional.softmax(output_student / T, dim=1)
            M = 0.5 * (P + Q)

            P = torch.clamp(P, 0.01, 0.99)
            M = torch.clamp(M, 0.01, 0.99)
            eps = 0.0
            loss_verifier_cig = 0.5 * F.kl_div(torch.log(P + eps), M) + 0.5 * F.kl_div(
                torch.log(Q + eps), M
            )
            # JS criteria - 0 means full correlation, 1 - means completely different
            loss_verifier_cig = 1.0 - torch.clamp(loss_verifier_cig, 0.0, 1.0)
            loss += loss_verifier_cig

        return loss, {'loss': loss.item()}

    def __repr__(self) -> str:
        return 'multi-model output kl loss'
