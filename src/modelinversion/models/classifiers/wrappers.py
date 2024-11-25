from copy import deepcopy
from typing import Iterator

from torch import Tensor
from torch.nn import MaxPool2d
from torch.nn.parameter import Parameter
from ...utils import (
    BaseHook,
    FirstInputHook,
    OutputHook,
    DeepInversionBNFeatureHook,
    traverse_module,
)
from .base import *


class BaseClassifierWrapper(BaseImageClassifier):

    def __init__(
        self,
        module: BaseImageClassifier,
        register_last_feature_hook=False,
    ) -> None:
        super().__init__(
            module.resolution,
            module.feature_dim,
            module.num_classes,
            register_last_feature_hook,
        )

        self.module = module

    def preprocess_config_before_save(self, config):
        # return config
        process_config = {}
        for k, v in config.items():
            if k != 'module':
                process_config[k] = v

        config['module'] = {
            'model_name': CLASSNAME_TO_NAME_MAPPING[self.module.__class__.__name__],
            'config': self.module.preprocess_config_before_save(
                self.module._config_mixin_dict
            ),
        }

        return super().preprocess_config_before_save(config)

    @staticmethod
    def postprocess_config_after_load(config):
        config['module'] = auto_classifier_from_pretrained(config['module'])
        return config


@register_model('vib')
class VibWrapper(BaseClassifierWrapper):

    @ModelMixin.register_to_config_init
    def __init__(
        self,
        module: BaseImageClassifier,
        register_last_feature_hook=False,
    ) -> None:
        super().__init__(
            module,
            # module.resolution,
            # module.feature_dim,
            # module.num_classes,
            register_last_feature_hook,
        )

        # self.module = module
        self.hidden_dim = module.feature_dim
        self.output_dim = module.num_classes
        self.k = self.hidden_dim // 2
        self.st_layer = nn.Linear(self.hidden_dim, self.k * 2)
        # operate_fc(self.module, self.k * 2, None)
        self.fc_layer = nn.Linear(self.k, module.num_classes)

        # self.feature_hook = FirstInputHook(self.fc_layer)

    # def get_last_feature_hook(self) -> BaseHook:
    #     return self.feature_hook

    @staticmethod
    def postprocess_config_after_load(config):
        config['module'] = auto_classifier_from_pretrained(
            config['module'], register_last_feature_hook=True
        )
        return config

    def _forward_impl(self, image: torch.Tensor, *args, **kwargs):

        # self._inner_hook.clear_feature()
        _, hook_res = self.module(image, *args, **kwargs)

        # # self._check_hook(HOOK_NAME_FEATURE)

        feature = hook_res[HOOK_NAME_FEATURE]

        statics = self.st_layer(feature)

        # statics, _ = self.module(image, *args, **kwargs)

        mu, std = statics[:, : self.k], statics[:, self.k : self.k * 2]

        self._last_statics = mu, std

        std = F.softplus(std - 5, beta=1)

        # eps = torch.FloatTensor(std.size()).normal_().to(std)
        eps = torch.randn_like(std)
        feat = mu + std * eps
        out = self.fc_layer(feat)

        return out, {'mu': mu, 'std': std, HOOK_NAME_FEATURE: feat}


def get_default_create_hidden_hook_fn(num: int = 3):

    param_num = num

    def _fn(model: BaseImageClassifier):
        linear_modules = []

        def _visit_fn(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                linear_modules.append(module)

        traverse_module(model, _visit_fn)
        linear_modules = linear_modules[1:]

        num = min(param_num, len(linear_modules))
        splitnum = (len(linear_modules) + 1) // (num + 1)
        use_nums = [splitnum * (i + 1) - 1 for i in range(num)]
        use_linear_modules = [linear_modules[i] for i in use_nums]
        return [FirstInputHook(l) for l in use_linear_modules]

    return _fn


def origin_vgg16_64_hidden_hook_fn(module):
    hiddens_hooks = []

    def _add_hook_fn(module):
        if isinstance(module, MaxPool2d):
            hiddens_hooks.append(OutputHook(module))

    traverse_module(module, _add_hook_fn, call_middle=False)
    return hiddens_hooks


@register_model('bido')
class BiDOWrapper(BaseClassifierWrapper):

    @ModelMixin.register_to_config_init
    def __init__(
        self,
        module: BaseImageClassifier,
        register_last_feature_hook=False,
        create_hidden_hook_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            module,
            # module.resolution,
            # module.feature_dim,
            # module.num_classes,
            register_last_feature_hook,
        )

        # self.module = module

        create_hidden_hook_fn = (
            create_hidden_hook_fn
            if create_hidden_hook_fn is not None
            else get_default_create_hidden_hook_fn()
        )

        self.hidden_hooks = create_hidden_hook_fn(self.module)
        print(f'hidden hook num: {len(self.hidden_hooks)}')
        # exit()

    # def get_last_feature_hook(self) -> BaseHook:
    #     return self.module.get_last_feature_hook()

    def _forward_impl(self, image: Tensor, *args, **kwargs):
        forward_res, addition_info = self.module(image, *args, **kwargs)
        addition_info[HOOK_NAME_HIDDEN] = [h.get_feature() for h in self.hidden_hooks]
        return forward_res, addition_info


def get_default_deepinversion_bn_hook_fn(num: int = 3):

    def _fn(model: BaseImageClassifier):
        bn_modules = []
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_modules.append(m)
        return [DeepInversionBNFeatureHook(l) for l in bn_modules]

    return _fn


class DeepInversionWrapper(BaseImageClassifier):

    def __init__(
        self,
        module: BaseImageClassifier,
        register_last_feature_hook=False,
        create_bn_hook_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            module.resolution,
            module.feature_dim,
            module.num_classes,
            register_last_feature_hook,
        )

        self.module = module

        create_bn_hook_fn = (
            create_bn_hook_fn
            if create_bn_hook_fn is not None
            else get_default_deepinversion_bn_hook_fn()
        )

        self.bn_hooks = create_bn_hook_fn(module)

    def get_last_feature_hook(self) -> BaseHook:
        return self.module.get_last_feature_hook()

    def _forward_impl(self, image: Tensor, *args, **kwargs):
        forward_res, addition_info = self.module(image, *args, **kwargs)
        addition_info[HOOK_NAME_DEEPINVERSION_BN] = [
            h.get_feature() for h in self.bn_hooks
        ]
        return forward_res, addition_info


class _ConditionMLP1d(nn.Module):

    def __init__(
        self,
        in_features,
        cond_features_num,
        cond_features_dim,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features + cond_features_dim, hidden_features)

        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        # self.drop = nn.Dropout(drop)

        self.cond_embedding = nn.Embedding(cond_features_num, cond_features_dim)

    def forward(self, x, cond):
        ori_x = x
        cond = self.cond_embedding(cond)
        x = torch.cat([x, cond], dim=-1)
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x) + ori_x
        # x = self.drop(x)
        return x


@register_model('cond_purifier')
class ConditionPurifierWrapper(BaseClassifierWrapper):

    @ModelMixin.register_to_config_init
    def __init__(
        self,
        module: BaseImageClassifier,
        cond_features_dim=512,
        register_last_feature_hook=False,
    ) -> None:
        super().__init__(
            module,
            register_last_feature_hook,
        )

        self.module.eval()

        for p in self.module.parameters():
            p.requires_grad = False

        self.purifier = _ConditionMLP1d(
            in_features=self.module.num_classes,
            cond_features_num=self.module.num_classes,
            cond_features_dim=cond_features_dim,
            hidden_features=cond_features_dim // 2,
            out_features=self.module.feature_dim,
            act_layer=nn.ReLU,
        )

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.purifier.parameters(recurse=recurse)

    def train(self, mode: bool = True):
        return self.purifier.train(mode)

    def eval(self):
        return self.purifier.eval()

    def _forward_impl(self, image: Tensor, *args, **kwargs):
        forward_res, addition_info = self.module(image, *args, **kwargs)
        # addition_info[HOOK_NAME_HIDDEN] = [h.get_feature() for h in self.hidden_hooks]
        addition_info['ori_logits'] = forward_res
        cond = torch.argmax(forward_res, dim=-1)
        forward_res = self.purifier(forward_res, cond)
        return forward_res, addition_info

class Inversion(nn.Module):
    def __init__(self, nc, ngf, nz, truncation, c):
        super(Inversion, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.nz = nz
        self.truncation = truncation
        self.c = c

        self.decoder = nn.Sequential(
            # input is Z
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0),
            nn.BatchNorm2d(ngf * 8),
            nn.Tanh(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.Tanh(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.Tanh(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.Tanh(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        topk, indices = torch.topk(x, self.truncation)
        topk = torch.clamp(torch.log(topk), min=-1000) + self.c
        topk_min = topk.min(1, keepdim=True)[0]
        topk = topk + F.relu(-topk_min)
        x = torch.zeros(len(x), self.nz).cuda().scatter_(1, indices, topk)

        x = x.view(-1, self.nz, 1, 1)
        x = self.decoder(x)
        x = x.view(-1, 1, 64, 64)
        return x

@register_model('adv_purifier')
class AdversarialPurifierWrapper(BaseClassifierWrapper):

    @ModelMixin.register_to_config_init
    def __init__(
        self,
        module: BaseImageClassifier,
        path: str,
        device: str, 
        iteration: int,
        eta: float,
        eps: float,
        register_last_feature_hook=False,
    ) -> None:
        super().__init__(
            module,
            register_last_feature_hook,
        )

        self.module.eval()
        self.iteration = iteration
        self.eta = eta
        self.eps = eps

        for p in self.module.parameters():
            p.requires_grad = False

        self.inversion = Inversion(nc=1, ngf=128, nz=530, truncation=530, c=50)
        self.inversion.load_state_dict(state_dict=torch.load(path, map_location='cpu'), strict=True)
        self.inversion.to(device=device)
        self.inversion.eval()
    
    def get_noise(self, ori_logit: Tensor, cur_logit: Tensor, grad: Tensor, eta: float, eps: float):
        logit = cur_logit + eta * torch.sign(grad)
        l = torch.argmax(ori_logit)
        m = torch.relu(torch.max(logit) - logit[l])
        logit[l] += m
        noise = logit - ori_logit
        if torch.norm(noise, p=1) > eps:
            noise *= (eps / torch.norm(noise, p=1))
        return noise
    
    

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.purifier.inversion(recurse=recurse)

    def _forward_impl(self, image: Tensor, *args, **kwargs):
        forward_res, addition_info = self.module(image, *args, **kwargs)
        # addition_info[HOOK_NAME_HIDDEN] = [h.get_feature() for h in self.hidden_hooks]
        addition_info['ori_logits'] = forward_res
        ori_logit = forward_res
        cur_logit = torch.tensor(forward_res, requires_grad=True)
        optimizer = torch.optim.Adam(cur_logit, lr=0.0002, betas=(0.5, 0.999), amsgrad=True)
        for i in range(self.iteration):
            recon = self.inversion(torch.softmax(cur_logit))
            loss = F.mse_loss(recon, image)
            loss.backward()
            cur_logit = ori_logit + self.get_noise(ori_logit=ori_logit, cur_logit=cur_logit, grad=loss.grad, eta=self.eta, eps=self.eps)
        return torch.softmax(cur_logit), addition_info
