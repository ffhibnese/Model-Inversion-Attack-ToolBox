import os
import json

import torch
from torch.nn import Module
from ...utils import ConfigMixin, safe_save


class ModelMixin(Module, ConfigMixin):

    # def save_config(self, save_path: str):
    #     os.makedirs(save_path, exist_ok=True)
    #     with open(save_path, 'w', encoding='utf8') as f:
    #         json.dump(f, self._config_mixin_dict)

    # @staticmethod
    # def load_config(config_path: str):
    #     if not os.path.exists(config_path):
    #         raise RuntimeError(f'config_path {config_path} is not existed.')

    #     with open(config_path, 'r', encoding='utf8') as f:
    #         kwargs = json.load(config_path)

    #     return kwargs

    def save_pretrained(self, path, **add_infos):
        save_result = {
            'state_dict': self.state_dict(),
            'config': self.preprocess_config_before_save(self._config_mixin_dict),
            **add_infos,
        }
        torch.save(save_result, path)

    @classmethod
    def from_pretrained(cls, data_or_path):

        if isinstance(data_or_path, str):
            data: dict = torch.load(data_or_path, map_location='cpu')
        else:
            data = data_or_path

        kwargs = cls.postprocess_config_after_load(data['config'])
        init_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        model = cls(**init_kwargs)

        if 'state_dict' in data:
            state_dict = data['state_dict']
            if state_dict is not None:
                model.load_state_dict(state_dict)

        return model
