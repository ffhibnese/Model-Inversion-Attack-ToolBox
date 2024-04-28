import os
import json
from ...utils import ConfigMixin


class ModelConfigMixin(ConfigMixin):

    def save_config(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        with open(save_path, 'w', encoding='utf8') as f:
            json.dump(f, self._config_mixin_dict)

    @classmethod
    def load_config(cls, config_path: str):
        if not os.path.exists(config_path):
            raise RuntimeError(f'config_path {config_path} is not existed.')

        with open(config_path, 'r', encoding='utf8') as f:
            kwargs = json.load(config_path)
        init_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        return cls(**init_kwargs)
