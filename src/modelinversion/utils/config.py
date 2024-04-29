import os
import inspect
import functools
import torch

from .io import safe_save


class ConfigMixin:

    def register_to_config(self, **config_dict):
        self._config_mixin_dict = config_dict

    def save_config(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        safe_save(self._config_mixin_dict, save_path)

    @classmethod
    def load_config(cls, config_path: str):
        if not os.path.exists(config_path):
            raise RuntimeError(f'config_path {config_path} is not existed.')

        kwargs = torch.load(config_path, map_location='cpu')
        init_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        return cls(**init_kwargs)

    @staticmethod
    def register_to_config_init(init):

        @functools.wraps(init)
        def inner_init(self, *args, **kwargs):
            # Ignore private kwargs in the init.
            init_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
            config_init_kwargs = {k: v for k, v in kwargs.items() if k.startswith("_")}
            if not isinstance(self, ConfigMixin):
                raise RuntimeError(
                    f"`@register_for_config` was applied to {self.__class__.__name__} init method, but this class does "
                    "not inherit from `ConfigMixin`."
                )

            # Get positional arguments aligned with kwargs
            new_kwargs = {}
            signature = inspect.signature(init)
            parameters = {
                name: p.default
                for i, (name, p) in enumerate(signature.parameters.items())
                if i > 0
            }
            for arg, name in zip(args, parameters.keys()):
                new_kwargs[name] = arg

            # Then add all kwargs
            new_kwargs.update(
                {
                    k: init_kwargs.get(k, default)
                    for k, default in parameters.items()
                    if k not in new_kwargs
                }
            )

            # Take note of the parameters that were not present in the loaded config
            if len(set(new_kwargs.keys()) - set(init_kwargs)) > 0:
                new_kwargs["_use_default_values"] = list(
                    set(new_kwargs.keys()) - set(init_kwargs)
                )

            new_kwargs = {**config_init_kwargs, **new_kwargs}
            getattr(self, "register_to_config")(**new_kwargs)
            init(self, *args, **init_kwargs)

        return inner_init
